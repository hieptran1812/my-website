---
title: "gRPC and Protocol Buffers: Contracts, Codegen, and Streaming"
date: "2026-06-20"
publishDate: "2026-06-20"
description: "A deep dive on the dominant high-performance RPC stack: Protobuf as the contract language and why field numbers, not names, are the wire identity; the binary wire format and where its speed actually comes from; codegen that makes the contract the code; gRPC over HTTP/2; the four streaming modes; deadlines, cancellation, status codes, and the browser gap — all on a Payments and Orders service."
tags:
  [
    "api-design",
    "api",
    "grpc",
    "protobuf",
    "rpc",
    "http2",
    "streaming",
    "codegen",
    "payments",
    "rest",
  ]
category: "software-development"
subcategory: "API Design"
author: "Hiep Tran"
featured: true
readTime: 57
image: "/imgs/blogs/grpc-and-protocol-buffers-contracts-codegen-and-streaming-1.png"
---

A payments team I worked with shipped a small "cleanup" to their internal Protobuf schema. The `PaymentRequest` message had a field named `amount` that everyone agreed was ambiguous — was it dollars or cents? So they renamed it `amount_minor` and, while they were in there, they deleted an old `legacy_currency` field that nobody used anymore. The new field reused the number the old one had been sitting on. The change looked clean in the diff. It passed code review. The Go service that owned `PaymentService` was redeployed in minutes.

Within the hour, the ledger reconciliation job — a Python service that still ran the *old* generated stub, because its deploy lagged by a day — started double-recording amounts and, worse, reading the freshly-renamed field's bytes through the *old* `legacy_currency` slot. The wire didn't carry the name `amount_minor` or `legacy_currency` at all. It carried a **field number**. And that number had quietly changed meaning. Two services now disagreed about what the same bytes *were*, and neither one threw an error, because Protobuf's whole design is to keep decoding even when it doesn't recognize something. The reconciliation totals drifted by tens of thousands of dollars before anyone caught it.

That incident is the entire post in one paragraph. gRPC and Protocol Buffers give you a fast, typed, polyglot, streaming RPC stack — and the single most important thing to understand about it is that the **contract's identity lives in field numbers, not field names**. Get that one rule right and everything else (the binary wire format, the codegen, the streaming modes, the deadlines) falls into place. Get it wrong and you reintroduce, in a strongly-typed system, exactly the silent-corruption bug that strong typing was supposed to prevent.

By the end of this post you will be able to: write a `.proto` contract from scratch with the right scalar types, field numbers, `optional`/`repeated`/`map`/`oneof`/`enum`; explain *why* the binary wire format is compact and fast, with the size derivation done honestly rather than hand-waved; run codegen so the contract becomes typed client and server stubs in any language; choose between the four method types (unary, server-streaming, client-streaming, bidirectional) by counting messages; and wire up deadlines, cancellation, status codes, metadata, and interceptors. We will build a `PaymentService` with a unary `CreatePayment` and a server-streaming `WatchPayouts` and stress-test it against retries, timeouts, schema changes, and the browser. This sits squarely in the series spine: an API is a **contract** you evolve over years for callers you will never meet, and gRPC just makes that contract a compiled artifact. If you want the framing of *why* RPC is sometimes the right shape at all, read [RPC vs REST: when a procedure beats a resource](/blog/software-development/api-design/rpc-vs-rest-when-a-procedure-beats-a-resource) first; if you want the decision framework for the whole field, read [choosing a paradigm: REST vs gRPC vs GraphQL by force](/blog/software-development/api-design/choosing-a-paradigm-rest-vs-grpc-vs-graphql-by-force). Here we go deep on the gRPC stack itself.

![A vertical stack figure showing the five gRPC layers from the proto contract down through codegen, gRPC framing, Protobuf bytes, and HTTP/2 transport](/imgs/blogs/grpc-and-protocol-buffers-contracts-codegen-and-streaming-1.png)

The figure above is the map for the whole post. Read it top to bottom: a `.proto` contract is compiled by codegen into typed stubs; the stubs serialize messages to Protobuf bytes; gRPC frames those bytes; and HTTP/2 carries the frames over multiplexed streams. We will walk every layer, but keep the top one in mind the whole way — the `.proto` is the source of truth, and the field numbers in it are the wire's identity.

## 1. Protocol Buffers as the contract language

Before there is any gRPC, there is **Protocol Buffers** — usually shortened to *Protobuf* or *protobuf*. Protobuf is two things at once: a small **interface definition language (IDL)** for describing the shape of structured data, and a **binary serialization format** for putting that data on the wire. gRPC is the RPC framework that uses Protobuf as its default contract and payload format, but Protobuf stands on its own — plenty of systems use it for storage and messaging without gRPC anywhere in sight.

An IDL is just a language whose only job is to *describe a contract* — the names, types, and structure of the messages two systems will exchange — independent of any programming language. You write the contract once in the IDL, and a compiler turns it into native types for Go, Python, Java, TypeScript, Rust, and a dozen others. The contract is the single source of truth; the language-specific code is generated from it. That is the whole point: nobody hand-writes the serialization logic, so nobody can accidentally make Go and Python disagree about the bytes.

Here is a complete, realistic `.proto` for our running example — the Payments service:

```protobuf
syntax = "proto3";

package commerce.payments.v1;

option go_package = "github.com/acme/commerce/gen/payments/v1;paymentsv1";

import "google/protobuf/timestamp.proto";

// A single payment against an order.
message Payment {
  string id = 1;                 // server-assigned, e.g. "pay_01H..."
  string order_id = 2;           // the order this payment settles
  int64 amount_minor = 3;        // smallest currency unit, e.g. cents
  string currency = 4;           // ISO-4217, e.g. "USD"
  PaymentStatus status = 5;
  google.protobuf.Timestamp created_at = 6;
  map<string, string> metadata = 7;  // free-form key/value tags
}

enum PaymentStatus {
  PAYMENT_STATUS_UNSPECIFIED = 0;  // proto3 requires a zero default
  PAYMENT_STATUS_PENDING = 1;
  PAYMENT_STATUS_SUCCEEDED = 2;
  PAYMENT_STATUS_FAILED = 3;
}

message CreatePaymentRequest {
  string order_id = 1;
  int64 amount_minor = 2;
  string currency = 3;
  string idempotency_key = 4;       // safe-retry token, like the REST header
  optional string description = 5;  // explicitly optional in proto3
}

message CreatePaymentResponse {
  Payment payment = 1;
}

message WatchPayoutsRequest {
  string merchant_id = 1;
  google.protobuf.Timestamp since = 2;  // resume point for the stream
}

message Payout {
  string id = 1;
  int64 amount_minor = 2;
  string currency = 3;
  google.protobuf.Timestamp settled_at = 4;
}

service PaymentService {
  // Unary: one request in, one response out.
  rpc CreatePayment(CreatePaymentRequest) returns (CreatePaymentResponse);

  // Server-streaming: one request, a stream of payouts back.
  rpc WatchPayouts(WatchPayoutsRequest) returns (stream Payout);
}
```

That is a working contract. Let me take it apart piece by piece, because every keyword in it is a decision with compatibility consequences.

### syntax, package, and options

`syntax = "proto3"` declares the dialect. proto3 is the modern default; the older proto2 is still around in legacy systems but proto3 is what you should write today. The `package commerce.payments.v1` namespaces the messages so a `Payment` here never collides with a `Payment` in some other service's schema — and notice the `v1`: putting the major version in the package is the conventional way to version a Protobuf API, because it lets `v1` and `v2` coexist as genuinely separate types rather than as the same type you keep mutating. The `option go_package` and the other `option` lines steer code generation per language; they do not change the wire at all.

### message, scalar types, and field numbers

A `message` is a record — a named bag of typed fields, like a struct. Each field has three parts: a **type**, a **name**, and a **field number**. In `int64 amount_minor = 3`, the type is `int64`, the name is `amount_minor`, and the field number is `3`.

The scalar types are the primitives you build everything from. The ones you will use constantly:

| Protobuf type | Maps to | Notes |
| --- | --- | --- |
| `double` / `float` | 64- / 32-bit float | fixed width on the wire |
| `int32` / `int64` | signed integers | varint-encoded; cheap for small values |
| `uint32` / `uint64` | unsigned integers | varint-encoded |
| `sint32` / `sint64` | signed, zig-zag | use these for values that go negative often |
| `fixed64` / `fixed32` | always 8 / 4 bytes | use when values are usually large |
| `bool` | boolean | one byte (a varint, value 0 or 1) |
| `string` | UTF-8 text | length-prefixed |
| `bytes` | raw byte blob | length-prefixed; no encoding assumed |

There is a real choice hiding in that table. `int64` is varint-encoded, which means small numbers cost few bytes and large numbers cost more — but negative numbers in plain `int64` always encode as ten bytes because of how two's-complement sign bits land in the varint. If a field is regularly negative, use `sint64`, which applies *zig-zag* encoding to map small-magnitude negatives to small varints. If a field is almost always a big number (a hash, a large id), `fixed64` is eight flat bytes and skips the varint math. These are micro-optimizations, but at a million payouts an hour they are real money in CPU and bandwidth.

Beyond the scalars, you will lean constantly on the **well-known types** — a small standard library of messages that ship with Protobuf, imported like `google/protobuf/timestamp.proto`. The ones that matter for a payments API: `google.protobuf.Timestamp` (a point in time as seconds-plus-nanos since the Unix epoch — use it instead of a raw string or int for `created_at`), `google.protobuf.Duration` (a span of time), `google.protobuf.Struct` (an arbitrary JSON-like object for genuinely dynamic data you cannot type ahead of time), and the wrapper types like `google.protobuf.Int64Value` and `google.protobuf.StringValue` (a boxed scalar that can be null — historically the way to get explicit presence before proto3 `optional` existed). Reaching for `Timestamp` rather than inventing your own time representation is not just convenience; it means every language's generated code maps it to the idiomatic local time type and every gRPC observer (loggers, tracers, gateways) already knows how to read it. The well-known types are a shared vocabulary, and using them keeps your contract legible to the whole ecosystem.

A word on **nesting and message composition**, because real schemas are not flat. A field's type can be another message — `Payment payment = 1` in `CreatePaymentResponse` embeds the whole `Payment` message. On the wire, an embedded message is just a length-delimited field (wire type 2): the tag, then the byte-length of the sub-message, then the sub-message's own (tag, value) pairs. This recursion is why one Protobuf decoder handles arbitrarily deep structures with the same tiny loop, and it is why a `repeated Payment` (a list of orders' payments, say) is cheap — each element is a length-delimited blob the decoder can skip or read without parsing a single field name. Composition is the right default: model your domain as small reusable messages (`Money`, `Address`, `Card`) and embed them, rather than flattening everything into one giant message with a hundred fields.

Now the crucial part. **The field number is the wire identity.** When Protobuf serializes `amount_minor = 4999`, it does not write the string `"amount_minor"` anywhere. It writes a tag derived from the number `3`, then the value. The name `amount_minor` exists only in your `.proto` and in the generated code; it never travels on the wire. This is the single most important fact about Protobuf, and it is exactly what bit the team in the opening story.

This is also the foundation of the compatibility rules, which I will state now and earn later:

- **Never reuse a field number.** If you delete a field, its number is poisoned forever. A new field on that number will be read by old code as the *old* field — silent corruption, no error.
- **Never renumber a field.** Changing `amount_minor = 3` to `amount_minor = 9` is, on the wire, deleting field 3 and adding field 9. Old clients write to 3 and read from 3; they will silently lose your data.
- **Renaming a field is free on the wire.** Because names never travel, renaming `amount` to `amount_minor` *without changing the number* changes nothing on the wire. (It does break source code that referenced the old name, so it is a code-level change, not a wire-level one — an important distinction.)
- **`reserved` is how you retire a number safely.** When you remove a field, reserve its number and name so no future engineer can accidentally bring them back.

The team's bug was three of these rules at once: they renumbered by deleting `legacy_currency` and putting `amount_minor` on its freed number, and they never reserved anything. The wire said "field N changed meaning" and Protobuf, doing exactly what it is designed to do, kept decoding.

### optional, repeated, map, oneof

Beyond a single scalar, fields have **cardinality**:

- **singular** (the default in proto3) — zero or one value. A singular scalar that is unset reads back as its zero value (`0`, `""`, `false`), and proto3 by default does not distinguish "set to zero" from "unset."
- **`optional`** — explicit presence. `optional string description = 5` lets you tell "the client sent an empty string" apart from "the client sent nothing." You need this whenever the default value is a meaningful business value (was the `amount` zero, or just absent?).
- **`repeated`** — a list. `repeated Payment payments = 1` is an ordered, possibly-empty list. On the wire, scalar `repeated` fields are *packed* by default in proto3 (all values under one tag), which is why a list of a thousand ints is far cheaper than a thousand separately-tagged fields.
- **`map<K, V>`** — an associative array. `map<string, string> metadata = 7` is sugar; on the wire it is encoded as a `repeated` message of key/value pairs, so it inherits all of `repeated`'s compatibility behavior.

`oneof` deserves its own callout. A `oneof` says "at most one of these fields is set at a time," and setting one clears the others. It is how you model a tagged union — a sum type — in Protobuf:

```protobuf
message PaymentMethod {
  oneof method {
    Card card = 1;
    BankTransfer bank_transfer = 2;
    Wallet wallet = 3;
  }
}
```

A `PaymentMethod` is a card *or* a bank transfer *or* a wallet, never two at once, and the generated code gives you a discriminated accessor to ask which one is present. Crucially, each member of a `oneof` still has its own field number, and those numbers still follow all the compatibility rules above. You can add a new member to a `oneof` (a new payment method) backward-compatibly, but you cannot move a field into or out of a `oneof` without a wire-level change.

### enum

An `enum` is a named set of integer constants. proto3 has one firm rule that trips up everyone the first time: **the first enum value must be zero, and it should be the "unspecified" sentinel.**

```protobuf
enum PaymentStatus {
  PAYMENT_STATUS_UNSPECIFIED = 0;  // mandatory zero default
  PAYMENT_STATUS_PENDING = 1;
  PAYMENT_STATUS_SUCCEEDED = 2;
  PAYMENT_STATUS_FAILED = 3;
}
```

The reason is the same presence problem: an unset enum field reads back as `0`, so if `0` meant `SUCCEEDED`, every message that forgot to set the status would silently claim success. Reserving `0` for `UNSPECIFIED` makes "I didn't set this" distinguishable from any real state. Enums are also forward-compatible in a specific, useful way: if a new server adds `PAYMENT_STATUS_REFUNDED = 4` and sends it to an old client that has never heard of value `4`, the old client keeps the raw number `4` around (in proto3 it is preserved as an unknown enum value) rather than crashing. Your client code must therefore always handle "a status I don't recognize" — usually by treating it as `UNSPECIFIED` and degrading gracefully.

## 2. Compatibility rules and the `reserved` keyword

I stated the rules; now let me make them rigorous, because they are the whole reason Protobuf is safe to evolve and the whole reason the opening incident happened.

The deep principle is the **tolerant reader**, the same robustness principle that governs safe REST evolution (covered in [backward and forward compatibility](/blog/software-development/api-design/backward-and-forward-compatibility-the-rules-of-safe-change)). A Protobuf decoder reads the stream tag by tag. For each tag it asks: do I know this field number? If yes, it decodes the value into the field. If *no*, it does not error — it skips the bytes (it can, because the tag's wire type tells it how many bytes to skip) and stashes them as *unknown fields*. This is precisely what makes additive change safe: a new field that an old reader has never heard of is simply ignored and (often) re-emitted untouched if the message is re-serialized.

That same mechanism is what makes number reuse catastrophic. The decoder trusts the number absolutely. If field 3 used to be `legacy_currency` (a string) and is now `amount_minor` (an `int64`), an old reader that still thinks 3 is a string will try to read your integer's bytes as a length-prefixed string — and either get garbage or, if the wire types happen to be compatible, a silently wrong value. There is no name to cross-check against. The number *is* the contract.

So the formal rules are:

1. **Adding a field is backward- and forward-compatible** as long as it uses a fresh, never-before-used number. Old readers skip it; new readers read it; everyone is fine.
2. **Removing a field is compatible only if you stop writing it and `reserved` its number.** Old readers that still expect it get the zero value, which your code should already tolerate (because singular fields can always be absent).
3. **Renaming a field is a no-op on the wire** but a breaking change in source code. Do it deliberately and separately.
4. **Renumbering or retyping a field is always breaking.** It is delete-plus-add on the wire.

This is where `reserved` earns its keep. After you remove a field, you reserve *both* its number and its name:

```protobuf
message Payment {
  reserved 8, 9, 12 to 15;       // poisoned field numbers
  reserved "legacy_currency", "old_status";  // poisoned names

  string id = 1;
  string order_id = 2;
  int64 amount_minor = 3;
  string currency = 4;
  PaymentStatus status = 5;
  google.protobuf.Timestamp created_at = 6;
  map<string, string> metadata = 7;
}
```

Now if any future engineer writes `int64 something = 8;` the compiler refuses to build — `8` is reserved. If they write `string legacy_currency = 20;` the compiler also refuses — that *name* is reserved. The reservation is a compile-time tripwire that makes the dangerous move impossible rather than merely discouraged. The opening incident never happens in a codebase that reserves religiously, because `protoc` would have rejected the renumbered field before it ever shipped.

#### Worked example: a `.proto` evolved safely with `reserved`

Suppose the Payments team genuinely needs to (a) rename `amount` to `amount_minor` for clarity, (b) remove the unused `legacy_currency`, and (c) add a new `risk_score`. Here is the safe version of the exact change that caused the incident.

The original message:

```protobuf
message Payment {
  string id = 1;
  string order_id = 2;
  int64 amount = 3;            // ambiguous name
  string currency = 4;
  string legacy_currency = 8;  // unused, to be removed
}
```

The safe evolution:

```protobuf
message Payment {
  reserved 8;                  // legacy_currency's number, poisoned
  reserved "legacy_currency";  // its name, also poisoned

  string id = 1;
  string order_id = 2;
  int64 amount_minor = 3;      // renamed in place — SAME number 3
  string currency = 4;
  double risk_score = 9;       // brand-new number, never used before
}
```

Walk the wire. The rename of `amount` to `amount_minor` keeps number `3`, so old and new code read and write the identical bytes — zero wire impact. The removal of `legacy_currency` stops writing field `8` and reserves it, so old readers that still expect `8` simply see it absent (zero value), and no future field can ever steal that number. The new `risk_score` takes number `9`, which has never been used, so old readers skip it as an unknown field and new readers read it. Every change is non-breaking. The contrast with the incident is the whole lesson: same business intent, but `amount_minor` stayed on `3` instead of stealing `8`, and `8` was reserved instead of recycled.

If you want the field-lifecycle playbook in full — the expand-contract migration recipes that pair the schema change with the rolling deploy — that is its own post: [schema evolution: adding, removing, renaming fields safely](/blog/software-development/api-design/schema-evolution-adding-removing-renaming-fields-safely). The `reserved` rule above is the Protobuf-specific core of it.

## 3. The binary wire format at a glance

Now the part everyone hears as "Protobuf is fast and small" and almost nobody can justify. Let me actually derive it, honestly, so you can explain *where* the win comes from and where it doesn't.

A Protobuf message on the wire is just a flat sequence of **(tag, value)** pairs, one per set field, in any order. The tag is a single varint that packs two things:

$$\text{tag} = (\text{field\_number} \ll 3) \mathbin{|} \text{wire\_type}$$

The low three bits are the **wire type** (which tells the decoder how to read the value); the rest is the field number. There are a handful of wire types: `0` for varint (ints, bools, enums), `1` for 64-bit fixed, `2` for length-delimited (strings, bytes, embedded messages, packed repeated), and `5` for 32-bit fixed. Three bits is enough because there are only a few. So for a field with number 2 of wire type 0, the tag byte is $(2 \ll 3) \mathbin{|} 0 = 16 = \texttt{0x10}$ — a single byte.

The value encoding depends on the wire type. The workhorse is the **varint** — a variable-length integer where each byte uses its low seven bits for data and its high bit as a "more bytes follow" continuation flag. So integers 0–127 take one byte, 128–16383 take two bytes, and so on. Small numbers are tiny; you only pay for the magnitude you actually use.

![A before and after figure contrasting the same payment payload as verbose JSON text versus compact Protobuf binary bytes](/imgs/blogs/grpc-and-protocol-buffers-contracts-codegen-and-streaming-2.png)

Let me make the size argument concrete by encoding one small payload both ways and counting bytes. Take a minimal payment: `amount_minor = 4999`, `currency = "USD"`, `status = SUCCEEDED (2)`. As JSON:

```json
{"amount_minor":4999,"currency":"USD","status":"SUCCEEDED"}
```

Count it: the braces, quotes, colons, and commas alone are structural overhead, and every field *name* is spelled out in full ASCII — `"amount_minor"` is fourteen bytes just for the key, before any value. The whole object is roughly 58 bytes, and that is a deliberately tiny example; real payments objects with nested order details and metadata routinely run several hundred bytes to a few kilobytes.

Now Protobuf. Field `amount_minor` is number 3, wire type 0 (varint): tag $(3 \ll 3)\mathbin{|}0 = 24 = \texttt{0x18}$, one byte. The value 4999 as a varint is two bytes ($4999 < 16384$). Field `currency` is number 4, wire type 2 (length-delimited): tag $(4 \ll 3)\mathbin{|}2 = 34 = \texttt{0x22}$, one byte, then a length byte `0x03`, then the three ASCII bytes `USD`. Field `status` is number 5, wire type 0: tag byte, plus the varint `2`, two bytes. Total: $1+2 + 1+1+3 + 1+1 = 10$ bytes. Against ~58 for JSON, that is roughly a 5–6× reduction *on this small object*, and the ratio stays favorable because the structural and field-name overhead JSON pays scales with every field and every message.

Be honest about where the win comes from, because it is not magic:

- **No field names on the wire.** Tags replace strings. This is the biggest single win and it grows with how many fields you have.
- **No structural punctuation.** No braces, quotes, colons, commas — length prefixes do that job in a byte or two.
- **Compact integers.** Varints mean small numbers cost almost nothing; JSON spells `4999` as four ASCII digits regardless.
- **Cheaper parsing.** Decoding is a tight loop of "read tag, switch on wire type, read N bytes" with no tokenizer, no string-to-number parsing of every digit, no UTF-8 quote-scanning. This is where the *speed* (not just the size) comes from — far less CPU per message and far less garbage allocated, which is what actually moves your p99 at scale.

And be honest about the caveats. Protobuf is **not self-describing**: those 10 bytes are meaningless without the `.proto`, whereas the JSON is human-readable on its own. Protobuf does **not** magically compress — if you gzip both, the gap narrows because gzip eats JSON's repetitive field names too (though Protobuf usually still wins, and you avoid the compression CPU). And for a payload that is mostly large free-text strings, the field-name savings are a smaller fraction of the total, so the ratio shrinks. The rule of thumb: Protobuf's advantage is largest for many-fielded, numeric, high-frequency messages, and smallest for a single big blob of text. For the deeper transfer-time-versus-payload-size argument, the [API performance](/blog/software-development/system-design/api-design-rest-grpc-graphql) discussion in the system-design track frames the latency side; here the takeaway is that the binary format buys you both fewer bytes and less CPU, and you can now say exactly why.

## 4. Codegen: the contract IS the code

Here is the property that makes the field-number discipline livable: you never write serialization by hand. You run a compiler over the `.proto` and it emits typed code.

The classic compiler is `protoc`, the Protobuf compiler, driven by per-language plugins (`protoc-gen-go`, `protoc-gen-go-grpc`, and so on). The modern, far friendlier front end is **buf**, which wraps `protoc`, manages your dependencies and plugins from a `buf.yaml`/`buf.gen.yaml`, lints your schema, and — the feature that would have prevented the opening incident outright — runs `buf breaking` to detect breaking changes against a baseline.

![A graph figure showing one PaymentService proto fanning out through the buf compiler into typed Go, Python, Java, and TypeScript stubs](/imgs/blogs/grpc-and-protocol-buffers-contracts-codegen-and-streaming-3.png)

The figure shows the shape: one `.proto` fans out to many language stubs, and because they are all generated from the same source, they cannot drift apart. Generating with buf is two short files and one command:

```yaml
# buf.gen.yaml
version: v2
plugins:
  - remote: buf.build/protocolbuffers/go
    out: gen
    opt: paths=source_relative
  - remote: buf.build/grpc/go
    out: gen
    opt: paths=source_relative
  - remote: buf.build/protocolbuffers/python
    out: gen
```

```bash
# Lint, check compatibility against main, then generate.
buf lint
buf breaking --against '.git#branch=main'
buf generate
```

That `buf breaking` step is the guardrail. If someone tries to renumber `amount_minor` from 3 to 9, or recycle a removed number, `buf breaking` fails the build with a precise message — the incident becomes a red CI check instead of a production page. This is the gRPC-native equivalent of the schema-diff and contract-testing discipline covered in [contract testing and schema diffs](/blog/software-development/api-design/contract-testing-consumer-driven-contracts-and-schema-diffs); for Protobuf, the linter *is* the contract test.

What does the generated code actually give you? On the server side, an interface (or abstract base class) with one method per `rpc` that you implement. On the client side, a fully-typed stub whose methods look like ordinary local function calls. Here is the Go server for `CreatePayment`:

```go
package payments

import (
	"context"

	paymentsv1 "github.com/acme/commerce/gen/payments/v1"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

type Server struct {
	paymentsv1.UnimplementedPaymentServiceServer
	store PaymentStore
}

func (s *Server) CreatePayment(
	ctx context.Context,
	req *paymentsv1.CreatePaymentRequest,
) (*paymentsv1.CreatePaymentResponse, error) {
	if req.GetAmountMinor() <= 0 {
		return nil, status.Errorf(
			codes.InvalidArgument, "amount_minor must be positive, got %d",
			req.GetAmountMinor())
	}
	// Idempotency: if we've seen this key, return the cached payment.
	if p, ok := s.store.ByIdempotencyKey(ctx, req.GetIdempotencyKey()); ok {
		return &paymentsv1.CreatePaymentResponse{Payment: p}, nil
	}
	p, err := s.store.Create(ctx, req)
	if err != nil {
		return nil, status.Errorf(codes.Internal, "create failed: %v", err)
	}
	return &paymentsv1.CreatePaymentResponse{Payment: p}, nil
}
```

And the Python client calling it:

```python
import grpc
from gen.payments.v1 import payments_pb2, payments_pb2_grpc

channel = grpc.insecure_channel("payments.internal:50051")
stub = payments_pb2_grpc.PaymentServiceStub(channel)

request = payments_pb2.CreatePaymentRequest(
    order_id="ord_8123",
    amount_minor=4999,
    currency="USD",
    idempotency_key="idem_9f2c-create-pay",
)
response = stub.CreatePayment(request, timeout=0.8)  # 800 ms deadline
print(response.payment.id, response.payment.status)
```

Notice what you did *not* write: no JSON, no `Content-Type`, no manual field mapping, no parsing. The contract is the code. The Python client's `CreatePaymentRequest` and the Go server's `*CreatePaymentRequest` are the same type, generated from the same `.proto`, and the compiler in each language enforces it. The `idempotency_key` field is just a contract field — gRPC has no built-in idempotency, so you carry the same safe-retry token you would in REST (the full pattern is in [idempotency keys and safe retries](/blog/software-development/api-design/idempotency-keys-safe-retries-and-exactly-once-illusions)) and implement it server-side, exactly as shown.

## 5. gRPC over HTTP/2: why the transport matters

gRPC does not invent its own transport — it runs over **HTTP/2**, and that choice is what enables streaming, multiplexing, and the low per-call overhead. To see why, you need to know what HTTP/2 fixed about HTTP/1.1.

Under HTTP/1.1, a TCP connection carries one request/response at a time. If you want concurrency you open many connections, and even then a slow response can block the ones queued behind it on the same connection (head-of-line blocking). HTTP/2 replaces this with a single connection carrying many independent, interleaved **streams**. Each stream is a bidirectional sequence of frames with its own id; the connection multiplexes frames from many streams together, so a slow call no longer blocks a fast one. HTTP/2 also uses **binary framing** (the protocol itself is binary, not text like HTTP/1.1) and **header compression** (HPACK), which matters because RPC traffic sends the same headers over and over.

The mapping from gRPC to HTTP/2 is clean:

- **One gRPC call is one HTTP/2 stream.** The method is encoded in the `:path` pseudo-header as `/commerce.payments.v1.PaymentService/CreatePayment`.
- **The request message(s)** travel as HTTP/2 DATA frames, each carrying one or more length-prefixed gRPC frames (the 1-byte compression flag plus the 4-byte big-endian length plus the Protobuf bytes).
- **Initial metadata** rides in HTTP/2 HEADERS frames at the start — this is where `authorization`, `grpc-timeout`, `content-type: application/grpc`, and your custom metadata live.
- **The status** comes back in **trailers** — HTTP/2 HEADERS frames sent *after* the response body, carrying `grpc-status` and `grpc-message`. Trailers are why gRPC can stream a response and still deliver an authoritative final status at the very end, after all the data has flowed.

Here is roughly what a single unary call looks like as HTTP/2 (rendered in the conventional pseudo-header notation):

```http
:method: POST
:scheme: https
:path: /commerce.payments.v1.PaymentService/CreatePayment
content-type: application/grpc+proto
te: trailers
grpc-timeout: 800m
authorization: Bearer <token>

<gRPC-framed Protobuf request bytes>

HTTP/2 200
content-type: application/grpc+proto

<gRPC-framed Protobuf response bytes>

grpc-status: 0
grpc-message:
```

That last block — the `grpc-status: 0` arriving as a trailer — is the gRPC contract's truth. The HTTP status is almost always `200`, even on application errors; the *real* outcome is the `grpc-status` trailer. This is a deliberate split and it is why you never read the HTTP status to decide if a gRPC call succeeded. Recall the layer map in figure 1: everything from the framing flag down through the Protobuf bytes rides inside these HTTP/2 DATA frames, and the status lives one layer up, in the trailers.

It is worth pausing on *why trailers exist at all*, because they are the unusual feature that makes streaming-plus-status work. In a normal HTTP/1.1 response, the status is the very first thing you send — the `200 OK` line — before any body. But a gRPC server-streaming call cannot know its final status until it has finished streaming: it might send forty payouts successfully and then hit a database error on the forty-first. If the status had to come first, you could never report a failure that happened *during* the stream. HTTP/2 trailers solve this exactly: the server streams the body, and only when it is done does it send the trailing HEADERS frame carrying `grpc-status` and `grpc-message`. The status is therefore always authoritative and always last, regardless of how much data preceded it. This is a small protocol detail with a large consequence — it is the reason gRPC can offer streaming and honest end-of-call errors at the same time, something a status-first protocol structurally cannot do.

There is one more subtlety in the framing itself. Each gRPC message inside the DATA frames is prefixed with a five-byte header: one byte of compression flag (is this message gzip-compressed?) and four big-endian bytes of length. That length prefix is what lets the receiver know exactly how many bytes to read for one message before the next one begins — essential for streaming, where many messages flow through the same byte stream. The compression flag is per-message, so a server can compress large messages and leave small ones uncompressed, paying CPU only where it saves bytes. None of this is something you implement; the generated stubs handle it. But knowing the frame is "flag, length, bytes, repeat" demystifies what you see when you capture gRPC traffic in a tool like Wireshark — it is not noise, it is a precise self-delimiting stream.

Because the connection is long-lived and multiplexed, the per-call overhead is tiny: no new TCP handshake, no new TLS handshake, compressed headers, binary frames. That is the other half of "gRPC is fast" — not just the compact Protobuf payload, but a transport built to reuse one connection for thousands of concurrent calls. And because streams are bidirectional by nature, the transport already supports every streaming mode for free; gRPC just exposes them. Contrast this with HTTP/1.1, where simulating a server-push feed meant long-polling or chunked transfer hacks, and bidirectional traffic meant WebSockets bolted on the side; HTTP/2's native streams make all four gRPC method types fall out of the same primitive rather than being special-cased.

One operational caveat that bites teams who treat the connection as free: HTTP/2 multiplexes streams over a *single* TCP connection, and a single connection is pinned to a single backend instance once a load balancer at layer 4 routes it. If you put a naive TCP load balancer in front of a gRPC fleet, every call from one client lands on one backend, and you lose balancing entirely. The fix is either a layer-7 (HTTP/2-aware) load balancer that balances per-*stream*, or client-side load balancing where the gRPC client itself spreads streams across resolved backends. This is exactly the kind of fleet concern covered in [service discovery and load balancing](/blog/software-development/microservices/service-discovery-and-load-balancing); the point for the contract here is that gRPC's connection model has real deployment consequences you must design for, not against.

## 6. The four method types

Streaming is where gRPC pulls decisively ahead of plain request/response REST, and the model is beautifully simple. A gRPC method is defined by two independent choices: is the **request** one message or a stream, and is the **response** one message or a stream? Two binary choices give four method types.

![A matrix figure laying out the four gRPC method types by request shape, response shape, and a use case for each](/imgs/blogs/grpc-and-protocol-buffers-contracts-codegen-and-streaming-4.png)

Here they are with their `.proto` signatures and when each one is the right tool:

| Method type | `.proto` signature shape | Request | Response | Reach for it when |
| --- | --- | --- | --- | --- |
| **Unary** | `rpc M(Req) returns (Resp)` | one | one | ordinary request/response: create a payment, fetch an order |
| **Server-streaming** | `rpc M(Req) returns (stream Resp)` | one | many | a feed from one query: live payouts, a tail of logs, search results paged as a stream |
| **Client-streaming** | `rpc M(stream Req) returns (Resp)` | many | one | uploading a large or open-ended sequence, then one summary: bulk import, metrics ingest |
| **Bidirectional** | `rpc M(stream Req) returns (stream Resp)` | many | many | a long-lived conversation: live chat, real-time sync, a control channel |

Our `PaymentService` uses two of these. `CreatePayment` is unary — one request, one response — because creating a payment is a single discrete transaction. `WatchPayouts` is server-streaming — one subscription request, then a stream of `Payout` messages as they settle — because the merchant wants a live feed without polling.

The two streaming directions you will reach for less often are still worth understanding. **Client-streaming** flips it: the client sends a stream and the server replies once at the end. Use it when the client has an open-ended or large sequence to push — say, streaming a million transaction rows for a bulk import — and only needs one summary back (rows accepted, errors found). **Bidirectional** opens both directions at once over the same stream; both sides read and write independently, which is what you want for a genuinely interactive session like a live operator chat or a real-time order-book sync.

To make client-streaming concrete, suppose the finance team wants to upload a day's transactions for reconciliation in one call rather than a million unary requests. The `.proto` adds:

```protobuf
service PaymentService {
  rpc CreatePayment(CreatePaymentRequest) returns (CreatePaymentResponse);
  rpc WatchPayouts(WatchPayoutsRequest) returns (stream Payout);
  // Client-streaming: a stream of rows in, one summary out.
  rpc ImportTransactions(stream TransactionRow) returns (ImportSummary);
}

message TransactionRow {
  string external_id = 1;
  int64 amount_minor = 2;
  string currency = 3;
}

message ImportSummary {
  int64 accepted = 1;
  int64 rejected = 2;
  repeated string errors = 3;
}
```

The client pushes rows and reads one summary at the end:

```python
def rows():
    for r in load_csv("2026-06-20.csv"):
        yield payments_pb2.TransactionRow(
            external_id=r.id, amount_minor=r.cents, currency=r.ccy)

summary = stub.ImportTransactions(rows(), timeout=120.0)
print(summary.accepted, summary.rejected, summary.errors)
```

The win is one connection, one deadline, backpressure if the server falls behind, and one round-trip's worth of metadata instead of a million. The same shape in bidirectional form — both sides streaming — is what powers a live operator chat: the client streams keystrokes while the server streams the agent's replies, both over the single open stream, neither blocking the other.

There is a memory implication worth stating plainly. Streaming exists precisely so you do *not* materialize the whole dataset in memory. A unary `ImportTransactions` carrying a million rows in one message would blow past the default 4 MB message limit and, even if you raised the limit, would force a giant allocation on both ends. The client-streaming version processes one row at a time on each side — bounded memory regardless of total size. When someone asks "how do I send a payload bigger than the message limit?", the gRPC-native answer is almost always "make it a stream," not "raise the limit."

A subtle but important point about ordering: within a single stream, messages are delivered **in order** — HTTP/2 guarantees ordering within a stream. But across *different* calls (different streams), there is no ordering guarantee, exactly as you would expect from concurrent requests. So `WatchPayouts` delivers payouts in the order the server writes them on that one stream, but two separate `WatchPayouts` subscriptions are independent.

#### Worked example: a server-streaming RPC with a deadline

Let me build `WatchPayouts` end to end, with a deadline, because this is where the streaming and the timeout machinery meet and where a naive implementation leaks goroutines and connections.

The server streams payouts as they arrive, checking on each one whether the client is still there:

```go
func (s *Server) WatchPayouts(
	req *paymentsv1.WatchPayoutsRequest,
	stream paymentsv1.PaymentService_WatchPayoutsServer,
) error {
	ctx := stream.Context() // carries the client's deadline + cancellation
	payouts, err := s.store.SubscribePayouts(ctx, req.GetMerchantId(), req.GetSince())
	if err != nil {
		return status.Errorf(codes.Internal, "subscribe failed: %v", err)
	}
	for {
		select {
		case <-ctx.Done():
			// Client cancelled or the deadline passed: stop cleanly.
			return status.FromContextError(ctx.Err()).Err()
		case p, ok := <-payouts:
			if !ok {
				return nil // upstream closed: end the stream with OK
			}
			if err := stream.Send(p); err != nil {
				return err // client went away mid-stream
			}
		}
	}
}
```

The Python client consumes the stream with a deadline on the whole subscription:

```python
import grpc
from gen.payments.v1 import payments_pb2, payments_pb2_grpc

stub = payments_pb2_grpc.PaymentServiceStub(channel)
request = payments_pb2.WatchPayoutsRequest(merchant_id="mer_42")

try:
    # timeout applies to the entire stream, not each message.
    for payout in stub.WatchPayouts(request, timeout=30.0):
        print(payout.id, payout.amount_minor, payout.currency)
except grpc.RpcError as e:
    if e.code() == grpc.StatusCode.DEADLINE_EXCEEDED:
        print("stream window elapsed; resubscribe with a fresh `since`")
    else:
        raise
```

Walk the failure modes, because this is the stress test that matters. If the client's 30-second deadline passes, the *server's* `ctx.Done()` fires (the deadline propagated down the stream), the loop exits, the subscription is released, and the client sees `DEADLINE_EXCEEDED` — no leaked goroutine, no orphaned database cursor. If the client process dies or calls cancel, the same `ctx.Done()` path fires server-side. If the client is slow to read and the server is fast to write, HTTP/2 flow control applies backpressure so the server's `Send` blocks rather than buffering unboundedly. The single most common bug in hand-rolled streaming is forgetting to watch `ctx.Done()`: the server keeps pushing into a dead stream and leaks resources. The `select` on `ctx.Done()` is not optional.

## 7. Deadlines, cancellation, and status codes

The single most consequential operational difference between gRPC and ad-hoc HTTP clients is that gRPC makes **deadlines first-class and propagating**. A deadline is an *absolute* point in wall-clock time by which the call must complete — not a per-hop timeout, but a budget for the whole operation that travels with the call.

![A timeline figure tracing a unary call under a deadline as the budget propagates from the client through the server and the database to a final status](/imgs/blogs/grpc-and-protocol-buffers-contracts-codegen-and-streaming-5.png)

Why absolute time, and why propagating? Consider a call chain: a gateway calls `PaymentService.CreatePayment`, which calls `LedgerService.Post`, which queries the database. If the client sets an 800 ms deadline, gRPC encodes it as the `grpc-timeout` header. The `PaymentService` receives a context that already knows "you have ~780 ms left" (some of the 800 spent in transit). When it calls `LedgerService`, it passes that same context, and `LedgerService` sees the *remaining* budget — say 620 ms — not a fresh 800. Every hop spends from the same wall clock. The instant the budget is gone, *every* in-flight stage gets a cancellation signal and fails fast with `DEADLINE_EXCEEDED` rather than each layer hanging on its own independent timeout. This is how you avoid the classic distributed pile-up where the client gave up long ago but five downstream services are still grinding on a request nobody is waiting for.

**Cancellation** is the same mechanism without the clock. If the client cancels (the user closed the tab, the parent call already failed), the cancellation propagates down the same context chain, and well-behaved servers stop work immediately. In Go this is `ctx.Done()`; in Python it surfaces as a cancelled `RpcError`; in every language the rule is the same — check for cancellation in any loop or long operation and bail out.

Then there are the **status codes**. gRPC defines its own canonical status code set, deliberately separate from HTTP status codes, returned in the `grpc-status` trailer. The full set is small and meaningful. Here is how the common ones map against the HTTP codes you already know:

| gRPC status (code) | Meaning | Closest HTTP | When you return it |
| --- | --- | --- | --- |
| `OK` (0) | success | 200 | the call succeeded |
| `INVALID_ARGUMENT` (3) | client sent bad input | 400 | `amount_minor <= 0`, malformed request |
| `NOT_FOUND` (5) | resource does not exist | 404 | no payment with that id |
| `ALREADY_EXISTS` (6) | resource already exists | 409 | duplicate create without idempotency |
| `PERMISSION_DENIED` (7) | authenticated but not allowed | 403 | wrong scope for this merchant |
| `UNAUTHENTICATED` (16) | missing/invalid credentials | 401 | no or bad bearer token |
| `RESOURCE_EXHAUSTED` (8) | quota or rate limit hit | 429 | over the rate limit |
| `FAILED_PRECONDITION` (9) | system state forbids this | 412 / 422 | order not in a payable state |
| `ABORTED` (10) | concurrency conflict | 409 | optimistic-lock collision |
| `DEADLINE_EXCEEDED` (4) | ran out of time budget | 504 | the deadline passed |
| `UNAVAILABLE` (14) | transient, retry it | 503 | server down, connection dropped |
| `INTERNAL` (13) | server bug | 500 | unexpected server-side failure |

Two of these carry real operational weight. `UNAVAILABLE` is the explicit "this is transient, you may safely retry" signal — gRPC clients use it to drive automatic retry-with-backoff, so returning it correctly (rather than a blanket `INTERNAL`) is what lets retries work. `DEADLINE_EXCEEDED` and `CANCELLED` (1) are how you distinguish "took too long" from "caller gave up" — different operational stories, different dashboards. The discipline mirrors honest HTTP status design (see [status codes that tell the truth](/blog/software-development/api-design/status-codes-that-tell-the-truth-2xx-3xx-4xx-5xx)): never collapse everything to `INTERNAL`, because the code *is* the machine-readable contract that tells the caller whether to retry, fix their input, or escalate.

## 8. The error model, metadata, and interceptors

A bare status code is rarely enough. "INVALID_ARGUMENT" does not tell the caller *which* argument. gRPC's richer error model uses `google.rpc.Status` — a message with a `code`, a human `message`, and a `details` list of arbitrary typed Protobuf messages. The standard `error_details.proto` ships well-known detail types: `BadRequest` (with per-field violations), `RetryInfo` (when to retry), `QuotaFailure`, `ErrorInfo` (a machine-readable reason and domain), and more.

Here is a validation failure returned with structured details on the Go server:

```go
import (
	"google.golang.org/genproto/googleapis/rpc/errdetails"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

func invalidAmount(got int64) error {
	st := status.New(codes.InvalidArgument, "amount_minor must be positive")
	br := &errdetails.BadRequest{
		FieldViolations: []*errdetails.BadRequest_FieldViolation{{
			Field:       "amount_minor",
			Description: fmt.Sprintf("must be > 0, got %d", got),
		}},
	}
	stWithDetails, err := st.WithDetails(br)
	if err != nil {
		return st.Err() // fall back to the plain status
	}
	return stWithDetails.Err()
}
```

The client can pull the typed `BadRequest` back out and react per-field, which is the gRPC analogue of the `problem+json` envelope from [error design](/blog/software-development/api-design/error-design-a-machine-readable-human-friendly-contract) — same goal (machine-readable *and* human-friendly errors), different serialization (typed Protobuf details instead of a JSON object).

**Metadata** is gRPC's term for what HTTP calls headers — key/value pairs sent alongside the message, not part of it. It carries `authorization` (the bearer token), trace and correlation ids, the `idempotency-key`, tenant ids, and anything cross-cutting. Keys ending in `-bin` carry binary values; everything else is ASCII. Metadata comes in two flavors: *initial metadata* (sent before the body, in HEADERS frames) and *trailing metadata* (sent after, in trailers — where `grpc-status` lives).

**Interceptors** are gRPC's middleware — functions that wrap every call, on the client or server side, to do cross-cutting work without touching each handler. A server interceptor is the single best place to enforce auth, emit RED metrics (rate, errors, duration), inject tracing, and log:

```go
func authInterceptor(
	ctx context.Context,
	req any,
	info *grpc.UnaryServerInfo,
	handler grpc.UnaryHandler,
) (any, error) {
	md, _ := metadata.FromIncomingContext(ctx)
	tokens := md.Get("authorization")
	if len(tokens) == 0 || !validBearer(tokens[0]) {
		return nil, status.Error(codes.Unauthenticated, "missing or invalid token")
	}
	// authenticated: proceed to the real handler.
	return handler(ctx, req)
}
```

Register that once and every unary RPC on the server is authenticated — no per-handler boilerplate, no handler that forgot to check. That centralization is exactly the gateway-style cross-cutting concern from [API gateways and the BFF pattern](/blog/software-development/api-design/api-gateways-routing-auth-rate-limiting-and-the-bff-pattern), implemented in-process. Streaming RPCs get a parallel `StreamServerInterceptor`.

Client interceptors are the mirror image: they wrap every outbound call to inject the auth token, propagate the trace context, and apply retry policy. The pattern that keeps a fleet observable is a matched pair — a client interceptor that stamps a correlation id into metadata on the way out, and a server interceptor that reads it back in and threads it through every log line and downstream call. Because metadata propagates, that one correlation id can follow a request through five services, which is the foundation of distributed tracing for a gRPC fleet (the broader observability picture, RED metrics and SLOs, is its own discipline). The discipline to internalize: cross-cutting behavior belongs in interceptors, never copied into handlers, because a handler that forgets to authenticate or forgets to log is a hole no amount of code review reliably catches.

## 9. Channels, connection management, and retries

A subtlety that trips up people coming from REST: in gRPC you do not open a connection per call. You create a long-lived **channel** — a wrapper over one or more HTTP/2 connections to a service — once, at startup, and share it for the life of the process. Every stub is built on a channel, and every call multiplexes onto the channel's existing connections. Creating a channel per request is the single most common gRPC performance mistake; it throws away the entire point of HTTP/2's connection reuse and pays a TCP-plus-TLS handshake on every call. Create the channel once; reuse it everywhere.

```python
# Do this once, at process startup — NOT per request.
channel = grpc.secure_channel(
    "payments.internal:50051",
    grpc.ssl_channel_credentials(),
    options=[
        ("grpc.keepalive_time_ms", 30000),       # ping idle connections
        ("grpc.max_receive_message_length", 8 * 1024 * 1024),
    ],
)
stub = payments_pb2_grpc.PaymentServiceStub(channel)
```

The channel also owns **retry policy**, and this is where gRPC's status-code discipline pays off concretely. gRPC supports a declarative retry config (a *service config*, usually delivered via the name resolver or pinned on the channel) that says, in effect: "retry up to 4 times, with exponential backoff, but *only* on `UNAVAILABLE`." That last clause is the crux. Retries are safe only for calls that are either idempotent or returned a status that proves no work happened. `UNAVAILABLE` (the server never received or never processed the call) is safe to retry; `INTERNAL` (the server might have done half the work) is not, unless you carry an idempotency key. This is the gRPC-native version of the safe-retry reasoning from [idempotency keys and safe retries](/blog/software-development/api-design/idempotency-keys-safe-retries-and-exactly-once-illusions): the status code tells the client whether retrying is safe, and an idempotency key makes retrying safe even when the status is ambiguous.

```json
{
  "methodConfig": [{
    "name": [{"service": "commerce.payments.v1.PaymentService"}],
    "retryPolicy": {
      "maxAttempts": 4,
      "initialBackoff": "0.1s",
      "maxBackoff": "2s",
      "backoffMultiplier": 2,
      "retryableStatusCodes": ["UNAVAILABLE"]
    }
  }]
}
```

#### Worked example: a retry that does not double-charge

Trace a `CreatePayment` that times out at the network layer. The client sends the request with `idempotency_key: "idem_9f2c"` and an 800 ms deadline. The request reaches the server, the server creates the payment and commits it to the database — but the response is lost on the way back (a transient network blip), so the client sees a timeout, not the `201`-equivalent success.

The naive client retries blindly and creates a second payment: the customer is charged twice. The correct client retries *with the same idempotency key*. This time the server's handler (from section 4) checks `s.store.ByIdempotencyKey(ctx, "idem_9f2c")`, finds the payment it created on the first attempt, and returns *that* cached payment with status `OK` — no second charge. The customer is charged exactly once despite two network round-trips. The deadline decided *when* the client gave up waiting; the idempotency key made the subsequent retry *safe*; and the gRPC status (`OK` on the retry, carrying the original payment) closed the loop. Note what gRPC did *not* do for you: there is no built-in exactly-once. You composed it from a deadline, a retry policy scoped to safe statuses, and an application-level idempotency key. "Exactly-once" on the wire is always an illusion built from at-least-once delivery plus idempotent handling — exactly as the message-queue track explains in [delivery semantics: at-most, at-least, exactly-once](/blog/software-development/message-queue/delivery-semantics-at-most-at-least-exactly-once).

## 10. gRPC-Web and the browser gap

Now the honest limitation that decides most architecture debates: **a browser cannot speak native gRPC.** The reason is concrete and not going away soon — gRPC requires fine-grained control over HTTP/2 frames (it reads trailers, manages framing, controls the stream lifecycle), and the browser `fetch` and `XMLHttpRequest` APIs do not expose that level of the HTTP/2 connection. JavaScript in a browser simply cannot construct a raw gRPC request.

The answer is **gRPC-Web**, a variant of the protocol that browsers *can* speak (it uses request/response shapes the `fetch` API supports and encodes trailers into the response body instead of real HTTP/2 trailers), plus a **proxy** that translates between gRPC-Web and native gRPC. The proxy — commonly Envoy with its `grpc_web` filter, or a built-in translator — sits between the browser and your backend.

![A graph figure showing a browser reaching a native gRPC PaymentService only through an Envoy gRPC-Web proxy while mobile and internal peers connect natively](/imgs/blogs/grpc-and-protocol-buffers-contracts-codegen-and-streaming-7.png)

As the figure shows, native callers (mobile apps, other backend services) reach `PaymentService` directly, while the browser goes through the proxy. The Envoy configuration is small:

```yaml
http_filters:
  - name: envoy.filters.http.grpc_web
  - name: envoy.filters.http.cors
  - name: envoy.filters.http.router
```

There is one more limitation worth knowing: gRPC-Web has historically supported **unary and server-streaming** well, but **client-streaming and bidirectional** streaming from the browser are limited or unsupported depending on the proxy and library. So a browser can do `CreatePayment` (unary) and `WatchPayouts` (server-streaming) through the proxy, but a browser-driven bidirectional chat over gRPC-Web is not a safe default. This is a major input to the build-versus-buy decision: if your primary client is a browser and you need full streaming, the proxy hop plus the streaming limits often tip the scales toward REST + SSE/WebSocket or GraphQL for that surface, while you keep gRPC for the internal fleet.

## 11. When gRPC shines and when it hurts

Time for the decisive part. gRPC is a sharp tool that is wrong for a surprising number of jobs people reach for it on. Choose by force, not fashion.

![A matrix figure comparing gRPC with Protobuf against REST with JSON across wire size, latency, browser reach, HTTP caching, and debuggability](/imgs/blogs/grpc-and-protocol-buffers-contracts-codegen-and-streaming-6.png)

The comparison above is the heart of the decision, and the pattern is clean: gRPC wins on the internal-performance axes (wire size, latency, polyglot type-safety, streaming) and loses on the public-surface axes (browser reach, HTTP caching, human debuggability). Walk each one.

**Where gRPC shines:**

- **Internal service-to-service traffic.** This is the home run. A fleet of microservices calling each other thousands of times a second benefits from every gRPC strength at once: compact Protobuf payloads, multiplexed HTTP/2 connections, typed stubs that make a wire mismatch a compile error, and propagating deadlines that stop the cascade pile-up. If you are building the kind of service mesh covered in [service-to-service security with mTLS](/blog/software-development/microservices/service-to-service-security-mtls-and-zero-trust), gRPC is the natural fit.
- **Low-latency, high-throughput paths.** The reduced bytes-and-CPU per call is real and shows up in the p99 at volume.
- **Polyglot organizations.** When the payments team writes Go, the ledger team writes Java, and the data team writes Python, codegen from one `.proto` keeps all three in lockstep with zero hand-written glue.
- **Streaming.** First-class server, client, and bidirectional streaming over a single connection, with backpressure and ordering, is something REST simply does not offer cleanly.

**Where gRPC hurts:**

- **Public and browser-facing APIs.** The browser gap means a proxy hop, and the streaming limits cap what a browser can do. For a public API where third-party developers will integrate, the friction of "you need our proto files and a generated client" is a real adoption tax that a plain JSON REST API does not impose.
- **Human debuggability.** You cannot `curl` a gRPC call and read the response; the bytes are opaque without the schema. Tooling like `grpcurl` and reflection help, but `curl https://api/orders/123` and reading the JSON is unbeatable for a 3 a.m. incident.
- **HTTP caching.** REST's `GET` + `ETag` + `Cache-Control` + CDN story has no clean gRPC equivalent. gRPC calls are `POST`s with binary bodies; intermediaries cannot cache them. If your read path is cache-friendly, REST keeps that win.
- **Simple CRUD with one client.** If you have one web frontend doing basic create/read/update/delete, gRPC's codegen and proxy overhead buys you complexity you do not need. Use REST.

![A tree figure splitting the gRPC decision into where it shines for internal streaming polyglot traffic versus where it hurts for public browser-facing cacheable APIs](/imgs/blogs/grpc-and-protocol-buffers-contracts-codegen-and-streaming-8.png)

The clean rule the tree above encodes: **gRPC for the inside, REST (or GraphQL) for the edge.** A very common and very good architecture is gRPC for all internal service-to-service calls plus a thin REST or GraphQL gateway at the public edge that translates to gRPC behind it — you get gRPC's performance internally and REST's reach and debuggability at the boundary. The full force-based comparison across all three paradigms is in [choosing a paradigm by force](/blog/software-development/api-design/choosing-a-paradigm-rest-vs-grpc-vs-graphql-by-force), and the system-design view at scale is in [API design: REST, gRPC, GraphQL](/blog/software-development/system-design/api-design-rest-grpc-graphql) — read both for the wider decision; this post owns the gRPC mechanics.

## 12. Case studies: gRPC in the real world

A few accurate, named references to ground the ideas above.

**gRPC at Google.** gRPC grew out of Google's internal RPC system, Stubby, which has carried Google's inter-service traffic for years. Google open-sourced gRPC in 2015 as a CNCF-hosted reimplementation of those ideas on HTTP/2 and Protobuf. Google's public **API Improvement Proposals (AIPs)** codify how they design APIs, including resource-oriented design that maps cleanly onto both gRPC and REST — and Google publishes gRPC transcoding so a single gRPC service can also be exposed as a REST/JSON API via annotations. The lesson worth taking: even Google, the originator, exposes a REST front for the public edge while running gRPC internally.

**Cloud-native infrastructure.** gRPC is the lingua franca of the CNCF ecosystem. Kubernetes components, etcd, containerd, Envoy's xDS configuration protocol, and a long list of CNCF projects use gRPC for their control-plane APIs. The reason is exactly the "internal, polyglot, low-latency, streaming" profile above — control planes are service-to-service traffic where typed contracts and streaming (etcd's watch API is a server stream of changes) are worth far more than browser reach. etcd's `Watch` is a textbook server-streaming RPC, structurally identical to our `WatchPayouts`.

**Netflix and large fleets.** Netflix and many other large microservice operators run gRPC for internal service-to-service communication where the volume and latency demands reward the binary protocol, while exposing REST or GraphQL at the device-facing edge (Netflix's edge has famously leaned on a GraphQL/BFF layer). Again the pattern is consistent: gRPC inside, a friendlier paradigm at the boundary.

**buf and the schema registry.** The buf toolchain and the **Buf Schema Registry (BSR)** treat `.proto` files as first-class, versioned, dependency-managed artifacts — you publish your schema, depend on others' schemas by module, and `buf breaking` enforces compatibility in CI. This is the operationalization of everything in section 2 and 4: the field-number discipline stops being tribal knowledge and becomes an enforced gate. An organization that adopts a schema registry plus `buf breaking` makes the opening incident structurally impossible.

**The Protobuf compatibility rules themselves** are documented in Google's official Protobuf language guide and are treated as a hard contract across Google's codebase: never reuse a tag number, reserve removed fields, the first enum value is zero. These are not style preferences — they are the safety rules of a system where the field number is the wire identity.

**gRPC transcoding and the both-worlds pattern.** Google publishes `grpc-gateway` and built-in HTTP/JSON transcoding (driven by `google.api.http` annotations in the `.proto`) that auto-generate a REST+JSON facade in front of a gRPC service. You annotate a method — `option (google.api.http) = { post: "/v1/payments" body: "*" };` — and a generated reverse proxy turns `POST /v1/payments` with a JSON body into a `CreatePayment` gRPC call, translating the response back to JSON. This is the operationalized version of "gRPC inside, REST at the edge": one `.proto` is the source of truth for *both* the internal binary contract and the public JSON API, so they cannot drift. It is the single most pragmatic way to get gRPC's internal benefits without imposing Protobuf on external integrators, and it is why many teams who "use REST" are in fact running a transcoded gRPC service underneath. The lesson threading every case study is the same: the schema is the asset, and once it is the asset, you can project it onto whatever surface each client needs.

## 13. Stress-testing the design

Let me close the loop by stress-testing `PaymentService` against the failure modes the series cares about, because a contract is only as good as its behavior under stress.

**What happens when the client retries on a timeout?** A unary `CreatePayment` that times out is ambiguous — did the server create the payment before the deadline passed, or not? gRPC has no built-in idempotency, so a naive retry double-charges. The fix is the same one REST uses: the `idempotency_key` field in `CreatePaymentRequest`, which the server uses to return the cached result on a retry rather than creating a second payment (the handler in section 4 does exactly this). The deadline tells you *when* to give up; the idempotency key makes giving-up-and-retrying *safe*.

**What happens when two writers race?** Two `CreatePayment` calls for the same order arrive concurrently. With an idempotency key, the second one returns the first one's result. Without one, you need a uniqueness constraint and you return `ALREADY_EXISTS` (gRPC code 6) — the honest signal that the resource is already there, the analogue of HTTP `409`.

**What happens when a field must be removed?** Section 2's answer: stop writing it, `reserve` its number and name, and old readers see the zero value they already tolerate. The wire stays stable; `buf breaking` enforces it; no client 500s.

**What happens when the payload is 10× bigger than planned?** Protobuf's binary encoding gives you headroom JSON does not — the same 10× growth costs fewer absolute bytes and far less parse CPU. But there are limits: gRPC enforces a default maximum message size (commonly 4 MB) precisely so a runaway message cannot exhaust memory. Past that, you stream — turn a giant single message into a server- or client-streaming RPC so the data flows in bounded chunks with backpressure rather than one enormous allocation. This is a place gRPC's streaming modes earn their keep: the answer to "the payload got too big" is often "make it a stream."

**What happens when a partner pins to an old schema for three years?** The package-versioned `commerce.payments.v1` coexists with a future `commerce.payments.v2` as separate types. The partner keeps calling `v1`; you keep `v1` running with its own service registration while `v2` evolves independently. The field-number discipline within `v1` means you can keep *additively* improving `v1` (new optional fields, new enum values, new `oneof` members) without ever breaking that partner — which is the whole promise of the series spine: change without breaking, for a caller you will never meet.

## When to reach for this (and when not to)

A decisive section, because the trade-offs are real and the wrong default is expensive.

**Reach for gRPC when:** you are building internal service-to-service APIs in a fleet; you need low latency and high throughput; you have a polyglot organization and want typed stubs everywhere from one contract; you need real streaming (live feeds, bulk ingest, bidirectional sessions); you control both ends and can run codegen on both. This is the strong default for the inside of a system.

**Do not reach for gRPC when:** your primary client is a browser and you need full streaming without a proxy hop — the browser gap and gRPC-Web's streaming limits make REST + SSE/WebSocket simpler. Do not use gRPC for a *public* API aimed at third-party developers who expect to `curl` it and read JSON — the "fetch our proto and generate a client" tax hurts adoption. Do not use gRPC where HTTP caching and CDN offload are central to your read path — gRPC's `POST`-of-binary calls are not cacheable. Do not reach for gRPC for a simple single-client CRUD app — you buy codegen, a proxy, and opaque debugging for performance you do not need.

There is also an organizational cost worth naming honestly: gRPC requires *both ends to run codegen*, which means a shared build step, a place to publish schemas, and a team culture that treats `.proto` changes as contract changes. That investment pays off across a large polyglot fleet, but for a two-person startup with one service and one frontend it is overhead with no payoff yet — start with REST and JSON, and adopt gRPC when the fleet, the latency budget, or the polyglot pressure actually arrives. Adopting it early "to be ready" usually just means debugging opaque binary streams while you still have ten endpoints. The honest rule is the same one the whole series preaches: choose the contract's machinery by the force you actually face, not by what looks modern.

The mature answer is rarely "gRPC everywhere" or "REST everywhere." It is **gRPC inside, a friendlier paradigm at the edge** — the architecture every case study above converged on independently. Choose per surface, by force.

## Key takeaways

- **The field number is the wire identity, not the field name.** Names never travel; tags do. This single fact drives every compatibility rule.
- **Never reuse or renumber a field; `reserve` removed numbers and names.** A recycled number is silent corruption an old reader cannot detect. The `reserved` keyword is a compile-time tripwire that makes the dangerous move impossible.
- **Protobuf is small and fast for a derivable reason:** no field names on the wire, no structural punctuation, varint integers, and a tokenizer-free decode loop. The advantage is largest for many-fielded numeric high-frequency messages and smallest for a single big text blob.
- **Codegen makes the contract the code.** One `.proto` compiles to typed stubs in every language; they cannot drift. `buf breaking` in CI turns the opening incident into a red check.
- **gRPC rides HTTP/2,** which gives multiplexed streams, binary framing, header compression, and trailers — the trailer is where `grpc-status` lives, so never read the HTTP status to judge a gRPC call.
- **Four method types are the cross product of single-or-stream on each side:** unary, server-streaming, client-streaming, bidirectional. Pick by counting messages.
- **Deadlines are absolute and propagating;** a budget travels with the call so the whole chain fails fast as `DEADLINE_EXCEEDED` instead of piling up. Always watch for cancellation in streaming loops or you leak resources.
- **gRPC status codes are the machine-readable contract:** `UNAVAILABLE` means retry, `INVALID_ARGUMENT` means fix your input, `DEADLINE_EXCEEDED` means too slow. Never collapse everything to `INTERNAL`.
- **The browser cannot speak native gRPC;** it needs gRPC-Web plus a proxy, with limited client/bidi streaming. Choose gRPC for the inside, REST or GraphQL at the public edge.

## Further reading

- [Protocol Buffers language guide (proto3)](https://protobuf.dev/programming-guides/proto3/) — the canonical reference for `.proto` syntax, scalar types, field numbers, and the compatibility rules.
- [gRPC official documentation](https://grpc.io/docs/) — concepts, the four method types, deadlines, metadata, interceptors, and per-language guides.
- [RFC 9113: HTTP/2](https://www.rfc-editor.org/rfc/rfc9113.html) — the transport gRPC rides on; streams, framing, HPACK, and flow control.
- [buf documentation](https://buf.build/docs/) — modern Protobuf tooling, `buf lint`, `buf breaking`, and the Buf Schema Registry.
- [Google API Improvement Proposals (AIPs)](https://google.aip.dev/) — Google's resource-oriented API design guidance for gRPC and REST.
- Within this series: the intro hub [what is an API: the contract between systems](/blog/software-development/api-design/what-is-an-api-the-contract-between-systems); the capstone [the API design playbook](/blog/software-development/api-design/the-api-design-playbook-a-review-checklist-first-endpoint-to-v2); and the siblings [RPC vs REST](/blog/software-development/api-design/rpc-vs-rest-when-a-procedure-beats-a-resource), [choosing a paradigm by force](/blog/software-development/api-design/choosing-a-paradigm-rest-vs-grpc-vs-graphql-by-force), and [schema evolution](/blog/software-development/api-design/schema-evolution-adding-removing-renaming-fields-safely).
