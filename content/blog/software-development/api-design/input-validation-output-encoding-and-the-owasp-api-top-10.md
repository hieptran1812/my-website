---
title: "Input Validation, Output Encoding, and the OWASP API Security Top 10"
date: "2026-06-20"
publishDate: "2026-06-20"
description: "Treat every request as hostile: validate at the boundary, bind only the fields you mean to, parameterize every query, and walk the OWASP API Security Top 10 with a concrete fix for each."
tags:
  [
    "api-design",
    "api",
    "security",
    "owasp",
    "input-validation",
    "mass-assignment",
    "ssrf",
    "http",
  ]
category: "software-development"
subcategory: "API Design"
author: "Hiep Tran"
featured: true
readTime: 56
image: "/imgs/blogs/input-validation-output-encoding-and-the-owasp-api-top-10-1.png"
---

A payments engineer once shipped a perfectly ordinary endpoint. `PATCH /orders/{id}` let a customer rename their order, change the shipping note, fix a typo in the delivery address. The handler was three lines: load the order, copy the request body onto it, save. It passed code review. It passed the integration tests. It ran clean in production for four months.

Then a security researcher sent a single request. The body was the same shape the mobile app sent — except it carried one extra field the app never set: `"status": "paid"`. The handler, being three honest lines, copied that onto the order too. The order flipped to `paid` without a cent moving. The fulfilment job, which only ever read `status`, packed the goods and shipped them. The company ate the loss. No exotic exploit, no buffer overflow, no zero-day — just an endpoint that trusted the shape of the request because it had only ever seen friendly clients send the friendly shape.

That is the whole subject of this post. Almost everything that goes wrong at the API boundary is not a clever attack on your cryptography or your memory allocator. It is your own code trusting input it had no business trusting: binding a whole request body to a model, concatenating a query parameter into SQL, fetching a URL a client supplied, returning a field the caller should never see. The defenses are not glamorous. They are boring, repeatable, and they live in specific, nameable places in the request lifecycle. This is the defensive-coding capstone of the security track, so we will be concrete: a hostile request, the line of code that lets it win, and the line that stops it.

![a layered diagram of defense gates from size limit through schema validation authorization and business rules before the handler and store](/imgs/blogs/input-validation-output-encoding-and-the-owasp-api-top-10-1.png)

By the end you will be able to do five things. Validate input at the boundary against an allowlist schema and reject everything else with a clean `422`. Bind only the fields a client is allowed to set, so over-posting `is_admin` or `balance` does nothing. Parameterize every query so injection is structurally impossible. Encode output and set the headers that stop a JSON value from becoming an XSS payload downstream. And walk the OWASP API Security Top 10 (the 2023 edition) item by item, with one concrete Payments-or-Orders example and one fix for each — and see why most of the list is a design and authorization failure, not a magic trick. This builds directly on the security siblings in this series: [authentication](/blog/software-development/api-design/authentication-api-keys-sessions-jwt-and-mtls) answers *who are you*, [authorization](/blog/software-development/api-design/authorization-scopes-roles-and-resource-level-permissions) answers *what may you do*, [rate limiting](/blog/software-development/api-design/rate-limiting-quotas-and-abuse-protection) answers *how much may you do*, and this post answers *can I trust what you sent*. As always, an API is a [contract between systems](/blog/software-development/api-design/what-is-an-api-the-contract-between-systems), and part of the contract is that you guarantee correctness even when the other side is lying.

## The principle: validate at the boundary, deny by default, fail closed

Before any technique, three rules. They are not slogans; each one is a structural property you can check in your code, and each one has a precise failure mode when you skip it.

**Validate at the boundary.** The boundary is the first place untrusted bytes become typed values your code reasons about — your request handler, your deserializer, your validation middleware. Validation that happens "somewhere deep in the service" is validation that some code path skipped, because deep code is reached by many paths and you will forget one. The principle: *a value is either validated at the edge or it does not enter the system.* Concretely, an order's `amount` is checked for type, sign, and range the instant the JSON is parsed — not in the pricing module that three handlers happen to share, one of which forgot to call it.

**Deny by default.** Every decision your input layer makes has a default. The default for "is this field allowed?" is *no*. The default for "is this content type acceptable?" is *no*. The default for "may this URL be fetched?" is *no*. You then carve out a small, explicit set of yeses — an *allowlist*. The opposite, a *denylist*, defaults to yes and tries to enumerate the noes, and it loses the moment a no appears that you did not think of. An allowlist that is too tight throws a `422` and someone files a ticket; a denylist that is too loose ships a breach. Asymmetric costs demand the safe default.

**Fail closed.** When validation cannot make a decision — the schema validator throws, the IP resolver times out, the canonicalizer hits input it cannot normalize — the answer is *reject*, not *allow*. "Fail closed" means the error path denies. The opposite, fail open, is the classic way a security control becomes decorative: the WAF that, when overloaded, passes traffic through unfiltered. In an API this is one line of discipline — your validation middleware's `catch` block returns `400`/`422`, it does not `next()` past the error.

These three compose into a pipeline you can draw and audit. A request crosses a series of gates, cheapest first, and any gate may reject it. The cheap gates (is the payload under the size cap? is the `Content-Type` one we accept?) run before the expensive ones (does this user own this object? does this satisfy the business invariant?), so a flood of garbage dies at the door instead of in your database.

![a flow graph where a raw request is canonicalized then checked against an allowlist and either passed to the handler or rejected with 422](/imgs/blogs/input-validation-output-encoding-and-the-owasp-api-top-10-2.png)

The quantitative version of "deny by default" is worth stating because it is the whole game. Suppose your input space has some set of inputs that are safe and a vastly larger set that is not — the unsafe set is effectively unbounded, because attackers invent new inputs. An allowlist of size $n$ accepts exactly those $n$ shapes and rejects the rest, so its false-accept rate against novel attacks is $0$ by construction. A denylist of size $m$ rejects $m$ known-bad shapes and accepts everything else, so its false-accept rate against a *novel* attack is $1$ — it has never seen it, so it lets it through. You do not maintain a list that has to grow forever to stay correct; you maintain a list that can only shrink as your API simplifies.

There is a corollary that catches people. "Fail closed" sounds obviously correct until you meet a scenario where it conflicts with availability, and then engineers quietly weaken it. Picture the payments team during an incident: the schema-validation service is overloaded and timing out, and orders are not getting through. The pressure is enormous to "just let requests pass when validation is slow, we'll re-check later." That is fail-open dressed as pragmatism, and it is exactly the moment an attacker is most likely to push malformed input through, because outages are noisy and nobody is reading the logs. The correct answer under load is to fail closed *fast* — return a `503` with `Retry-After` so clients back off — not to fail open and let unvalidated bodies reach your database. The whole point of failing closed is that it holds precisely when the system is degraded, which is when you can least afford to be lying to yourself about what input you trust. Build your validation so that its *failure* is a rejection, never a bypass, and load-test that the rejection path is cheap enough to survive a flood.

Why order the gates cheapest-first? Because the work you do before you reject is work an attacker gets for free. If a request must pass schema validation, then object authorization, then a business-rule check, and you run them in that order, a 50 MB garbage body is fully parsed and validated before the size check that should have killed it at the door. Reorder so the size check (a comparison against a counter the framework already keeps) runs before the parse, the content-type check (a string compare) runs before the deserializer, and the cheap authentication check runs before the expensive ownership query that hits the database. Each gate's *cost* and each gate's *rejection rate* under attack should both push it earlier: the cheaper it is and the more attack traffic it kills, the closer to the edge it belongs. A flood of unauthenticated junk should die at a string comparison, not at a database round-trip.

## Input validation: schema, allowlist, canonicalize

Input validation is the gate that turns "some JSON arrived" into "a well-formed Order-creation request, or a rejection." There are four moving parts, and the order matters.

### Validate against a schema, not against `if` statements

Hand-written validation drifts. One handler checks `amount > 0`, another forgot, a third checks `amount >= 0` and now you accept free orders. The fix is to declare the contract once, in a schema, and validate every request against it mechanically. If you publish an [OpenAPI](/blog/software-development/api-design/what-is-an-api-the-contract-between-systems) document, you already have most of a schema; you can enforce it at runtime with request-validation middleware so the spec and the runtime check can never disagree.

Here is a JSON Schema for creating an order on our Payments platform. It is the contract, and it is strict on purpose.

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "title": "CreateOrderRequest",
  "type": "object",
  "additionalProperties": false,
  "required": ["currency", "items"],
  "properties": {
    "currency": { "type": "string", "enum": ["USD", "EUR", "GBP"] },
    "note": { "type": "string", "maxLength": 280 },
    "items": {
      "type": "array",
      "minItems": 1,
      "maxItems": 50,
      "items": {
        "type": "object",
        "additionalProperties": false,
        "required": ["sku", "quantity"],
        "properties": {
          "sku": { "type": "string", "pattern": "^[A-Z0-9_-]{1,32}$" },
          "quantity": { "type": "integer", "minimum": 1, "maximum": 999 }
        }
      }
    }
  }
}
```

Read what this schema *refuses*. `additionalProperties: false` is the single most important line: any field the client sends that is not in the list is a hard error, not a silently-ignored extra. That one keyword is the difference between the broken `PATCH /orders/{id}` in the opening story and a safe one. `enum` on `currency` is an allowlist of three values — a fourth is rejected, no `if` chain required. `pattern` on `sku` is an allowlist of characters; `maxLength` on `note` and `maxItems` on `items` are the length and size bounds that keep a single request from ballooning into a denial of service. Notice there is no `status`, no `total`, no `is_admin` — those are not the client's to set, so they are simply absent from the schema, and `additionalProperties: false` turns any attempt to send them into a `422`.

The same idea in code, with a typed validator at the boundary:

```python
from pydantic import BaseModel, conint, constr, conlist, field_validator
from typing import Literal

class OrderItem(BaseModel):
    model_config = {"extra": "forbid"}  # reject unknown fields
    sku: constr(pattern=r"^[A-Z0-9_-]{1,32}$")
    quantity: conint(ge=1, le=999)

class CreateOrderRequest(BaseModel):
    model_config = {"extra": "forbid"}
    currency: Literal["USD", "EUR", "GBP"]
    note: constr(max_length=280) | None = None
    items: conlist(OrderItem, min_length=1, max_length=50)
```

`extra="forbid"` is the `additionalProperties: false` of the typed world. The `Literal`, the bounded ints, the bounded list — every one is an allowlist constraint expressed as a type. When the handler receives a `CreateOrderRequest`, every field in it has already been proven to satisfy the contract, and nothing else made it through.

### Allowlist, not denylist

The single most consequential choice in validation is the default. An allowlist says "here is exactly what is permitted; reject the rest." A denylist says "here is what is forbidden; permit the rest." They sound symmetric. They are not.

![a matrix comparing allowlist and denylist across default decision new input encoding tricks and maintenance](/imgs/blogs/input-validation-output-encoding-and-the-owasp-api-top-10-3.png)

Consider validating a `sku`. The denylist instinct is "reject anything with a quote, a semicolon, angle brackets" — the characters that show up in injection payloads. But a `sku` is `[A-Z0-9_-]`; an allowlist says exactly that, in four characters of regex, and it is correct forever. The denylist has to anticipate every dangerous character in every downstream context (the quote that breaks SQL, the angle bracket that breaks HTML, the newline that breaks a log line, the null byte that breaks a C library three services down), and it is wrong the moment a context you did not consider appears.

| Dimension | Allowlist (recommended) | Denylist |
| --- | --- | --- |
| Default decision | Deny; carve out yeses | Allow; carve out noes |
| New, unforeseen input | Rejected until explicitly added | Accepted until explicitly blocked |
| Correctness over time | Stays correct; list only shrinks | Decays; list must grow forever |
| Encoding bypasses | Closed if you canonicalize first | Open via any encoding you missed |
| Failure mode | A real request gets a `422` (ticket) | A malicious request gets through (breach) |
| Where it lives | Schema, types, enums, regex | Ad-hoc string scanning, WAF rules |

The asymmetry is the point. An over-tight allowlist produces a support ticket; an under-tight denylist produces an incident. Always pay in tickets.

### Type, range, length, and format checks

Schema validation gives you the four classic checks for free, but name them so you do not forget any:

- **Type.** `quantity` is an integer, not the string `"1"`, not `1.5`, not `true`. Loose languages coerce silently; a strict validator refuses, because the difference between `5` and `"5"` is the difference between arithmetic and string concatenation downstream.
- **Range.** `quantity` is `1..999`. `amount` is positive. A `page_size` is `1..100`. Unbounded numeric inputs are how a `?limit=` turns into an out-of-memory crash, which is exactly the resource-consumption risk we will meet in the Top 10.
- **Length.** `note` is `<= 280` chars; `items` is `<= 50`. Every string and every array gets a maximum, because an input without a length bound is a denial-of-service vector waiting for a 50 MB string.
- **Format.** `sku` matches a pattern; a `currency` is a known enum; an email matches an email shape; a UUID matches a UUID shape. Format checks are allowlists of structure.

### Reject unknown fields

Worth its own line because it is the one developers fight: yes, *reject* unknown fields, do not ignore them. The tolerant-reader principle — be liberal in what you accept — is correct for *evolution* (an old client should not break when the server adds an optional response field), but it is dangerous as a blanket rule for *requests*. The compromise the industry has settled on: be tolerant of unknown fields the *server* sends to clients (forward compatibility), and strict about unknown fields *clients* send to you (security). A client that sends `status` to `POST /orders` is either confused or hostile; either way you want to know, and a `422` that says "unknown field: status" is a kindness to the confused and a wall to the hostile.

### Canonicalize before you validate

The subtle one. Validation compares input against a rule, but the same logical input can have many byte representations: `%2e%2e%2f` and `../`, uppercase and lowercase, NFC and NFD Unicode, a path with `.` and `..` segments, a URL with userinfo and an `@`. If you validate the raw bytes and then *decode* before use, an attacker validates one form and you act on another — the classic validation bypass.

The rule: **canonicalize first, then validate the canonical form, then use the canonical form.** Decode percent-encoding, normalize Unicode, resolve path segments, lowercase the host — produce one canonical representation, validate *that*, and use *that*. Do not validate the raw and use the decoded; do not decode twice. For a URL you will fetch (the SSRF case below), canonicalization means resolving it all the way to an IP and validating the IP, because the hostname is not what your socket connects to.

## Mass assignment and over-posting

This is the bug from the opening story, and it deserves the most detailed treatment because it is the most common serious API vulnerability that pure input validation does *not* catch. You can validate every field's type and range perfectly and still be wide open, because the problem is not a bad value — it is binding a *field the client should not control at all.*

The mechanism: your handler receives a request body and binds the whole thing to your domain model or database row. Frameworks make this a one-liner — `Order(**body)`, `order.update(body)`, `entity.MergeFrom(req)` — and it is wonderful for the fields you meant. It is a catastrophe for the fields you did not, because the client controls the body, and the body can carry any field your model has. If your `Order` model has a `status`, a `total`, a `customer_id`, or an `is_admin`, and you bind the whole body, the client can set them.

![a before and after diagram contrasting binding the whole request body which lets a client set admin against binding a typed DTO which ignores it](/imgs/blogs/input-validation-output-encoding-and-the-owasp-api-top-10-4.png)

#### Worked example: setting an unintended field

Here is the vulnerable handler. It is the kind of code that ships because it is short and it works for the happy path.

```python
@app.patch("/orders/{order_id}")
def update_order(order_id: str, body: dict):
    order = db.orders.get(order_id)
    for key, value in body.items():     # bind the WHOLE body
        setattr(order, key, value)
    db.orders.save(order)
    return order
```

The intended use is a customer fixing the shipping note:

```http
PATCH /orders/ord_8812 HTTP/1.1
Host: api.example.com
Authorization: Bearer <token>
Content-Type: application/json

{ "note": "Please leave at the back door" }
```

That works fine. Now the attack — the same endpoint, the same auth, one extra field the model happens to have:

```http
PATCH /orders/ord_8812 HTTP/1.1
Host: api.example.com
Authorization: Bearer <token>
Content-Type: application/json

{ "note": "Please leave at the back door", "status": "paid", "total": 0 }
```

The loop sets `order.status = "paid"` and `order.total = 0`. The order is now marked paid for nothing. Swap in `"customer_id": "victim_acct"` and you reassign the order to another account; swap in `"is_admin": true` on a `/users/me` endpoint and you escalate to admin. The validator, if there is one, sees a string `status` and a number `total` and waves them through — they are valid *values*; they are just not *fields the client may set.*

#### Worked example: the allowlist / DTO fix

The fix is to bind only an explicit allowlist of fields. A Data Transfer Object (a DTO — a small input type that names exactly the writable fields, separate from your domain model) is the cleanest expression of this. The DTO has no `status`, no `total`, no `customer_id`, so they cannot be set no matter what the client sends.

```python
from pydantic import BaseModel, constr

class UpdateOrderDto(BaseModel):
    model_config = {"extra": "forbid"}   # unknown fields -> 422
    note: constr(max_length=280) | None = None
    shipping_address_id: str | None = None
    # NOTE: no status, no total, no customer_id, no is_admin

@app.patch("/orders/{order_id}")
def update_order(order_id: str, dto: UpdateOrderDto):
    order = db.orders.get(order_id)
    # only the DTO's fields can ever be applied
    update = dto.model_dump(exclude_unset=True)   # only fields the client sent
    for key, value in update.items():
        setattr(order, key, value)
    db.orders.save(order)
    return OrderResponse.from_order(order)        # DTO on the way out too
```

Now the malicious request gets a `422` ("unknown field: status, total"), because `UpdateOrderDto` forbids extras. Even if you preferred to silently ignore extras, the loop only ever sees `note` and `shipping_address_id`, because those are the only fields the DTO carries. `exclude_unset=True` adds a second nicety: it distinguishes "field omitted" from "field set to null," which is the correct semantics for `PATCH` (you only touch the fields the client actually sent).

The general rule, paradigm-independent: **never bind the request body directly to a persistence model.** Bind it to an input type whose fields are exactly the ones the client is allowed to write. In Rails this is strong parameters (`params.require(:order).permit(:note)`); in .NET it is a view model / `[Bind]` allowlist; in gRPC it is using a dedicated `UpdateOrderRequest` message plus a `FieldMask` so the server only applies named paths; in GraphQL it is a tight `input` type. The names differ; the discipline is identical — *write an allowlist of writable fields and bind only those.*

This is also why mass assignment shows up twice in the OWASP list. Setting a field you should not control is API3 (Broken Object Property Level Authorization) on the *write* side. The same property-level failure on the *read* side — returning a field the caller should not see — is the *excessive data exposure* half of API3, and the symmetric fix is a response DTO: serialize through a type that names exactly the fields this caller may see, never your raw model. We will return to it.

It is worth naming why the *convenient* default in every framework is the dangerous one. Frameworks are optimized for the demo: `Order(**body)` writes itself, the tutorial works in three lines, and the field-by-field DTO looks like boilerplate you can skip. The trap is that the convenient version is correct for exactly the fields you are thinking about while you write it — and silently wrong for every field you add to the model *later*. Six months on, someone adds an `is_internal` flag to the `Order` model for an admin feature, never touches the customer-facing handler, and has just made `is_internal` settable by any customer, because the handler binds the whole body and the model grew a new field. The DTO does not have this failure: a new model field is not in the DTO, so it is not bindable until someone deliberately adds it. The allowlist is *resilient to future change* in a way the full-body bind never is, and that resilience is the real reason to pay the boilerplate cost.

The discipline is paradigm-independent, but it looks different across the stack, so name the equivalent in whatever you ship:

- **REST with a typed framework** — a request DTO / input model with `extra="forbid"` (the example above), Rails strong parameters (`params.require(:order).permit(:note, :shipping_address_id)`), or .NET view models with an explicit `[Bind]` list.
- **gRPC / Protobuf** — do not reuse your storage message as the request message. Define a dedicated `UpdateOrderRequest` that carries an `Order` plus a `google.protobuf.FieldMask`, and on the server apply *only* the paths named in the mask, validated against an allowlist of mutable paths. A field mask is an allowlist of which fields this update may touch.
- **GraphQL** — define a tight `input UpdateOrderInput` type with only the writable fields, never accept your full domain type as input, and resolve each field through a resolver that checks authorization. GraphQL makes excessive *exposure* especially easy (a client can request any field in the schema), so the response side needs field-level authorization just as much as the input side needs a narrow input type.

```protobuf
message UpdateOrderRequest {
  string order_id = 1;
  Order order = 2;                       // the new values
  google.protobuf.FieldMask update_mask = 3;  // allowlist of paths to apply
}
// server applies ONLY paths in update_mask that are in the mutable allowlist:
//   note, shipping_address_id   (NOT status, total, customer_id)
```

**Stress test.** What happens when the field you forgot to exclude is one the client *legitimately* sets on a *different* endpoint? Say `status` is client-settable on `POST /orders/{id}/cancel` (the client may set `status: cancelled`) but not on `PATCH /orders/{id}`. The lesson is that "writable" is per-endpoint, not per-field: the cancel endpoint has its own DTO that allows exactly the `cancelled` transition (and validates it is a legal state change), while the generic patch endpoint's DTO does not mention `status` at all. Do not centralize a single "fields the client may write to Order" list; centralize *per-operation* DTOs, because the same field has different write rules in different operations. And what happens when a partner integration genuinely needs to set `customer_id` (an internal tool placing orders on behalf of a customer)? Then that is a *different, privileged* endpoint with its own DTO *and* an authorization check for the privileged scope — never a special case smuggled into the customer-facing handler. The shape of the fix scales: more privilege means a more privileged endpoint with a wider DTO gated by a stronger authorization check, never a wider DTO on the same endpoint everyone can reach.

## Injection: parameterize, never concatenate

Injection is the oldest API vulnerability and the easiest to eliminate completely, and it has nothing to do with filtering "bad characters." Injection happens when input crosses from *data* into *code* — when a value you received gets parsed as part of a command. The cure is to keep the boundary between code and data fixed, so input can never cross it. You do that by *parameterizing*: you write the query (the code) with placeholders, and you hand the values (the data) to the driver separately. The database engine compiles the query first, then binds the values as pure data; a value can never be reparsed as SQL because the parser already ran.

![a graph showing client filter input either concatenated into a query leading to injection or bound as a parameter staying as data](/imgs/blogs/input-validation-output-encoding-and-the-owasp-api-top-10-7.png)

The classic mistake, on a Payments search endpoint:

```python
# DANGER: input concatenated into the query string
def find_orders(customer_id: str):
    q = f"SELECT * FROM orders WHERE customer_id = '{customer_id}'"
    return db.execute(q)
```

A request of `GET /orders?customer_id=x' OR '1'='1` makes the query return every order in the table; `'; DROP TABLE orders; --` is the textbook escalation. The fix is one line different — the value never touches the query text:

```python
# SAFE: the value is bound as a parameter, never parsed as SQL
def find_orders(customer_id: str):
    q = "SELECT * FROM orders WHERE customer_id = %s"
    return db.execute(q, (customer_id,))
```

```sql
-- the engine sees this query shape, compiles it ONCE,
-- then binds the value as data; the value is never parsed as SQL
SELECT * FROM orders WHERE customer_id = $1;
```

The same principle generalizes to every interpreter input crosses:

- **NoSQL.** A MongoDB query built from a raw body lets `{"customer_id": {"$ne": null}}` match everything, or `{"$where": "..."}` run arbitrary JavaScript. Validate that `customer_id` is a string (not an object) at the boundary, and pass it as a value, not by splatting the body into the filter.
- **OS commands.** Never build a shell string from input. Use the array form of `exec` (`["convert", path, "-resize", size]`) with no shell, so input is an argument, never a command — and validate `path` against an allowlist first.
- **Headers / log / response splitting.** Input containing `\r\n` placed into a response header can inject a second header or split the response. Strip or reject control characters; never copy raw input into a header value.

The reason parameterization works where escaping fails is worth making precise, because "just escape the input" is the advice that keeps injection alive. Escaping tries to make a value *safe to embed in code* by transforming the dangerous characters; it is a denylist of characters dressed up, and it inherits every weakness of a denylist. You have to escape correctly for the specific dialect (MySQL, Postgres, and SQLite disagree on quoting), in the specific context (a string literal escapes differently than an identifier), and you have to never forget one site. Parameterization sidesteps the entire problem: the query text is sent to the engine and *compiled* with placeholders where the values go, so the parse tree is fixed before any value is bound. The value arrives afterward, as data, into a slot the parser already decided is a value — there is no character it could contain that would change the already-compiled structure. Escaping fights the parser; parameterization runs after it. That is why the rule is "parameterize," not "escape carefully."

A subtlety that trips people: parameterization binds *values*, not *identifiers*. You cannot parameterize a table name, a column name, or a sort direction — `ORDER BY $1` does not work, because those are part of the query structure, not data. This is exactly where validation and parameterization combine: the *value* in a `WHERE` clause is parameterized, but the *column* you sort by must be checked against an allowlist of permitted columns before it is interpolated, because it cannot be a parameter. So `?sort=created_at` becomes a lookup in a fixed map (`{"created_at": "orders.created_at", "amount": "orders.amount"}`) that returns the real, trusted column expression, and an unknown sort key is a `422`. Never interpolate a client-supplied identifier without an allowlist; never interpolate a client-supplied value at all.

There is an important link to make here: the safest place to handle the *structured* query inputs your API exposes — filter expressions, sort fields, sparse field selections — is a constrained, allowlisted layer, not raw passthrough to your database. This series covers exactly that pattern in [filtering, sorting, and sparse fieldsets without reinventing SQL](/blog/software-development/api-design/filtering-sorting-and-sparse-fieldsets-without-reinventing-sql): you accept `?sort=created_at` only if `created_at` is in an allowlist of sortable columns, you map filter operators to a fixed grammar, and the actual SQL is always parameterized. The API never lets the client write SQL; it lets the client choose from a menu you control. That is allowlisting applied to query expressiveness.

### When to validate and when to parameterize

These are different defenses for different jobs, and conflating them is how people get injection wrong. Validation rejects malformed input at the boundary; parameterization keeps well-formed input as data at the sink. You need both, and parameterization is the one that *prevents* injection — validation alone cannot, because a value can be perfectly valid and still be dangerous in the wrong context. A name like `O'Brien` is a legitimate value that breaks a concatenated query and is harmless in a parameterized one. So: validate to reject garbage early, parameterize to make injection structurally impossible. Never rely on validation or escaping to "clean" input for a query — parameterize.

## Output encoding: even a JSON API can feed an XSS

It is tempting to think output encoding is a browser concern, not an API one — "we return JSON, not HTML, so cross-site scripting (XSS, where attacker-controlled text becomes executable script in a victim's browser) is someone else's problem." That is half true and the dangerous half. The data your API returns is rendered *somewhere*: in a single-page app's DOM, in an admin dashboard, in a partner's web UI, in an internal tool that builds an HTML email from your response. If a value the attacker stored through your API (an order `note` of `<script>steal()</script>`) is later dropped into a page without encoding, your API was the delivery vehicle for stored XSS. The store accepted it; the render executed it.

So output handling for an API has two responsibilities. First, **encode per context** — the rule is that data is encoded for the context it is rendered in, and that is the *consumer's* job at render time (HTML-encode for HTML, attribute-encode for an attribute, JavaScript-encode for a script context). You cannot do the consumer's contextual encoding for them, because you do not know their context. But you can refuse to let your *response* be misinterpreted as something executable, and you can avoid storing data in forms that are dangerous by default.

Second, and squarely your job: **declare your content type honestly and forbid sniffing.** Two headers do almost all the work.

```http
HTTP/1.1 200 OK
Content-Type: application/json; charset=utf-8
X-Content-Type-Options: nosniff
Cache-Control: no-store
```

`Content-Type: application/json` tells the browser this is data, not a document to render. `X-Content-Type-Options: nosniff` tells the browser to *believe* the `Content-Type` and not "sniff" the bytes and decide for itself — without it, a browser that sees HTML-looking bytes may render your JSON as HTML and execute embedded script (a real attack class against endpoints that reflect input). Together they say: this is JSON, treat it as JSON, do not get clever. A JSON API that returns user-controlled content without `nosniff` has handed the browser permission to reinterpret it.

The security headers an API should generally send, with the reasoning:

| Header | Value | What it buys you |
| --- | --- | --- |
| `Content-Type` | `application/json; charset=utf-8` | Honest media type; the basis for everything else |
| `X-Content-Type-Options` | `nosniff` | Browser must not re-guess the type and execute it |
| `Strict-Transport-Security` | `max-age=63072000` | Force HTTPS; stop downgrade and cookie theft |
| `Cache-Control` | `no-store` (for sensitive responses) | Keep PII out of shared caches and disk |
| `Content-Security-Policy` | `default-src 'none'` (for API responses) | If a response *is* rendered, allow nothing |
| `Access-Control-Allow-Origin` | a specific origin, never `*` with credentials | Stop hostile sites reading authenticated responses |

A note on encoding the response body itself: do not pre-escape values *inside* your JSON (HTML-escaping a `note` field server-side produces `&lt;b&gt;` in the data, which is wrong for a non-HTML consumer and double-escapes for an HTML one). Return the true value as JSON, set the headers above, and let each consumer encode for its own context. The exception is data you *know* will be rendered as HTML you control — but at the API layer, store the truth and encode at render.

The principle behind contextual encoding is the same code-versus-data boundary as injection, but at the *output* sink instead of the *input* sink. A `note` of `<script>steal()</script>` is harmless *data* until it lands in a context that treats angle brackets as *code* — an HTML body, where it becomes a script tag; an HTML attribute, where a quote breaks out of the attribute; a JavaScript string, where a backtick or `</script>` breaks out of the string; a URL, where it can become a `javascript:` scheme. Each context has a different "dangerous character" set, which is exactly why the encoding must happen *at* the render context and cannot be done once at the API. The API's job is to not corrupt the data and to refuse to let its *own* response be misread as code; the consumer's job is to encode for wherever it renders. When you control both ends (your own SPA), you still encode at render — modern front-end frameworks do this by default when you bind text into the DOM, which is one more reason to bind text, never to build HTML strings by concatenation.

#### Worked example: a stored value becomes XSS

Trace the failure end to end so the API's role is unmistakable. A customer creates an order with a malicious note:

```http
POST /orders HTTP/1.1
Host: api.example.com
Authorization: Bearer <token>
Content-Type: application/json

{ "currency": "USD", "note": "<img src=x onerror=alert(document.cookie)>",
  "items": [ { "sku": "WIDGET_1", "quantity": 1 } ] }
```

The note is a *valid string* — it passes schema validation (it is under 280 chars, it is a string), and it should, because rejecting all angle brackets would break legitimate notes and is a denylist anyway. The order is stored with the true note. Later, an internal fulfilment dashboard lists pending orders and drops each note straight into the page with a string template: `"<td>" + order.note + "</td>"`. Now the `onerror` fires in the operator's authenticated browser session — stored XSS, delivered through your API, executing in your own staff's session. The API stored the truth (correct); the dashboard rendered without encoding (the bug). The fix is at the dashboard (HTML-encode `order.note` at render, or use a framework that does), and the API's contribution is to have set `Content-Type: application/json` and `nosniff` so the *API's own* responses are never executed, and to have stored the value faithfully rather than mangling it. The takeaway: input validation correctly accepts the string; output encoding at the render context is what makes it safe; the two controls are not interchangeable.

## SSRF: when a URL input becomes the attacker's request

Server-Side Request Forgery (SSRF) is the vulnerability where your server makes an HTTP request to a URL the *client* supplied, and the client points it somewhere it should not go. APIs are full of URL inputs: a webhook callback URL, an "import from URL" feature, an image-fetch by URL, an avatar-by-URL, an OpenAPI-spec-by-URL importer. Each is an invitation for the client to make *your server* the attacker, and your server is on the inside of your network.

![a before and after diagram contrasting fetching a raw webhook URL that hits cloud metadata against an allowlisted fetch that blocks internal ranges](/imgs/blogs/input-validation-output-encoding-and-the-owasp-api-top-10-5.png)

The crown-jewel target in the cloud is the instance metadata service, reachable at a link-local address from inside many environments, which hands out temporary credentials to anything that asks from the right place. If your server will fetch any URL a client gives it, the client gives it that address and your server fetches your cloud credentials and returns them. Other targets: internal admin panels with no auth (because "they're internal"), databases on private IPs, other services' health endpoints, and `localhost`.

#### Worked example: SSRF via a webhook URL

Our Payments API lets a merchant register a webhook so we POST them order events. The naive registration just stores whatever URL they send, and the delivery worker fetches it. Here is the unsafe path:

```python
@app.post("/webhooks")
def register_webhook(dto: WebhookDto):
    # stores any URL; the worker will later POST to it
    db.webhooks.save(merchant_id, dto.url)
    # some flows even verify it immediately:
    requests.post(dto.url, json={"type": "ping"})   # DANGER: fetches client URL
```

The attack is a registration whose URL is not a merchant endpoint at all:

```http
POST /webhooks HTTP/1.1
Host: api.example.com
Authorization: Bearer <token>
Content-Type: application/json

{ "url": "http://169.254.169.254/latest/meta-data/iam/security-credentials/" }
```

Your server dutifully fetches the cloud metadata endpoint and, depending on the flow, either returns the credentials in the verification response or leaks them through error messages or timing. The merchant supplied a URL; your server made the request; the request hit the inside.

#### Worked example: the allowlist and IP-block fix

The fix has three layers, and you need all three because each closes a bypass in the others.

```python
import ipaddress, socket
from urllib.parse import urlparse

ALLOWED_SCHEMES = {"https"}                 # 1. scheme allowlist: https only
BLOCKED_NETS = [
    ipaddress.ip_network("127.0.0.0/8"),    # loopback
    ipaddress.ip_network("10.0.0.0/8"),     # private
    ipaddress.ip_network("172.16.0.0/12"),  # private
    ipaddress.ip_network("192.168.0.0/16"), # private
    ipaddress.ip_network("169.254.0.0/16"), # link-local (metadata!)
    ipaddress.ip_network("::1/128"),        # ipv6 loopback
    ipaddress.ip_network("fc00::/7"),       # ipv6 unique-local
]

def assert_safe_url(raw: str) -> str:
    u = urlparse(raw)
    if u.scheme not in ALLOWED_SCHEMES:
        raise ValueError("scheme not allowed")
    # 2. resolve to the IP we will ACTUALLY connect to, then validate THAT
    infos = socket.getaddrinfo(u.hostname, u.port or 443)
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if any(ip in net for net in BLOCKED_NETS) or not ip.is_global:
            raise ValueError("resolves to a blocked address")
    return raw
```

The three layers:

1. **Scheme allowlist.** Only `https`. This kills `file://` (read local files), `gopher://` and `dict://` (talk to arbitrary TCP services), and plain `http`.
2. **Resolve, then validate the resolved IP.** You must check the IP your socket will actually connect to, not the hostname. A hostname like `metadata.attacker.com` can resolve to `169.254.169.254`. So resolve with `getaddrinfo`, then reject any address in a private, loopback, or link-local range — and reject anything that is not globally routable.
3. **Block the time-of-check/time-of-use gap.** A determined attacker resolves the host to a safe IP at validation time and a blocked IP at fetch time (DNS rebinding). The robust fix is to *connect to the validated IP directly* (pin the resolved address and set the `Host` header), or perform all outbound fetches through a dedicated egress proxy that enforces the allowlist at connect time. For most APIs, also disabling redirects (or re-validating every redirect hop) closes the easy version.

Operationally, the strongest control is architectural: route all client-driven outbound requests through an egress gateway on a network that simply *cannot* reach your internal ranges or the metadata endpoint. Then even a validation bug cannot reach the inside, because the route does not exist. This is the same zero-trust posture the microservices track covers in [service-to-service security with mTLS and zero trust](/blog/software-development/microservices/service-to-service-security-mtls-and-zero-trust) — assume the network is hostile and authenticate and constrain every hop, including your own server's outbound calls.

## Size and complexity limits: validation as DoS defense

A request can be perfectly well-typed and still be an attack, because parsing it costs resources. Every input that has no upper bound is a denial-of-service vector. The defenses are limits, and they belong at the boundary because by the time you have parsed a 100 MB JSON document to discover it is too big, you have already paid for it.

- **Payload size.** Cap the request body (e.g. 256 KB for a JSON API; higher only for explicit upload endpoints). Enforce it at the gateway *and* the framework, and return `413 Payload Too Large`. A body cap stops the 50 MB string and the billion-element array before your parser runs.
- **JSON nesting depth.** A deeply nested document — `[[[[[...]]]]]` thousands deep — can exhaust the stack of a recursive parser. Cap nesting depth (most mature parsers offer a setting) and reject beyond it.
- **Array and string length.** Already in the schema (`maxItems`, `maxLength`), but say it again at the parse layer: an unbounded array is unbounded work in any per-element loop, and an $O(n^2)$ operation on a million-element array is a one-request outage.
- **Key count and duplicate keys.** Cap the number of object keys; reject or canonicalize duplicate keys (different parsers resolve `{"a":1,"a":2}` differently, which is itself a validation-bypass surface).
- **Algorithmic-complexity inputs.** A regex with catastrophic backtracking applied to attacker input is a DoS (ReDoS). Use linear-time regex engines or simple anchored patterns for input validation, and never run an unbounded user-supplied pattern.

The general statement: every input dimension — bytes, depth, breadth, count — gets a maximum, and exceeding it is a fast, cheap rejection at the door. This is the same family of control as [rate limiting](/blog/software-development/api-design/rate-limiting-quotas-and-abuse-protection), which bounds *how many* requests; size limits bound *how big* one request is. Together they bound resource consumption, which is API4 in the list below.

## Walking the OWASP API Security Top 10 (2023)

The OWASP API Security Top 10 is a community-maintained list of the most critical API security risks, published by the Open Worldwide Application Security Project. The 2023 edition is the current one, and it is *API-specific* — distinct from the general web-application Top 10 — because APIs fail differently. The standout finding of the API list, across both its editions, is that the dominant risks are not injection or buffer overflows; they are **authorization** failures (an authenticated caller acting on data or functions that are not theirs) and **design** failures (an endpoint that does exactly what it was built to do, abused). Most of this list is fixed in your design and your authorization layer, not with a scanner.

![a matrix mapping selected OWASP API Top 10 risks to their primary defense showing authorization and limits cover most of the list](/imgs/blogs/input-validation-output-encoding-and-the-owasp-api-top-10-6.png)

A note on the 2023 changes, since you will see both editions referenced. The 2019 list's "Excessive Data Exposure" and "Mass Assignment" were *merged* in 2023 into **API3: Broken Object Property Level Authorization** — the recognition that returning a field you should not (read) and setting a field you should not (write) are the same property-level authorization failure. The 2019 "Lack of Resources and Rate Limiting" became the broader **API4: Unrestricted Resource Consumption**. And 2023 added two genuinely new entries that reflect how APIs actually get breached: **API6: Unrestricted Access to Sensitive Business Flows** (the business-logic abuse risk) and **API10: Unsafe Consumption of APIs** (trusting third parties you call). The list got more honest about authorization and business logic.

Here is the full list, each with a one-line risk, a concrete Payments/Orders example, and the primary defense.

| Risk (2023) | One-line risk | Concrete example (Payments/Orders) | Primary defense |
| --- | --- | --- | --- |
| API1 BOLA | Act on another user's object by ID | `GET /orders/ord_999` for someone else's order | Check the caller owns the object, per request |
| API2 Broken Authentication | Auth can be bypassed or forged | Unsigned JWT accepted; no token expiry | Strong, standard authN; verify every token |
| API3 Object Property Authz | Read/write a property you may not | Set `status:paid`; response leaks `card_pan` | DTO in and out; field-level allowlists |
| API4 Resource Consumption | One request burns unbounded resources | `?page_size=1000000`; 50 MB body | Size caps, pagination limits, rate limits |
| API5 Function-Level Authz | Call an admin function as a normal user | `DELETE /orders/ord_1` allowed for any user | Authorize the operation, not just the route |
| API6 Sensitive Business Flows | Automate a flow faster than humans should | Script claims every discount code in seconds | Detect and throttle the flow; bot defense |
| API7 SSRF | URL input reaches internal hosts | Webhook URL is the metadata endpoint | Scheme allowlist; block internal IP ranges |
| API8 Security Misconfiguration | Insecure defaults left in place | Verbose stack traces; CORS `*`; debug on | Hardened, reviewed config; least exposure |
| API9 Improper Inventory | Forgotten or undocumented endpoints | `/v1/orders` still live, unpatched, unmonitored | Inventory every API and version; retire old |
| API10 Unsafe 3rd-party | Trusting data from an API you call | Currency-rate API response stored unvalidated | Validate responses; sandbox and limit them |

Now each in turn, briefly but concretely.

### API1 — Broken Object Level Authorization (BOLA)

The most common and most damaging API vulnerability. The caller is authenticated, but the endpoint checks *that you are someone*, not *that this object is yours*. `GET /orders/{id}` loads the order by ID and returns it without checking the order belongs to you, so you increment the ID (or paste another customer's order ID) and read their order, their address, their card last-four. This is purely an authorization failure and validation cannot help — the ID is well-formed; it is just not yours.

The fix is to scope every object access to the caller: load the object *and* verify ownership in the same breath, ideally by making the query itself ownership-scoped (`WHERE id = $1 AND customer_id = $2`), so a mismatch returns `404` (not `403` — do not confirm the object exists). This is the heart of [authorization, scopes, roles, and resource-level permissions](/blog/software-development/api-design/authorization-scopes-roles-and-resource-level-permissions); BOLA is what happens when resource-level permission checks are missing.

### API2 — Broken Authentication

The mechanisms that establish *who you are* are weak or bypassable: JWTs accepted with `alg: none`, tokens that never expire, credential-stuffing with no lockout, password reset that leaks tokens, API keys in URLs that land in logs. A concrete one: an endpoint verifies the JWT signature but not the `exp` claim, so a token stolen a year ago still works. The defense is to use standard, vetted authentication and verify *every* property of a credential — signature, expiry, issuer, audience — on *every* request. The full treatment is in [authentication: API keys, sessions, JWT, and mTLS](/blog/software-development/api-design/authentication-api-keys-sessions-jwt-and-mtls); the rule here is that authentication is a gate, and a gate that does not check every property is a gate left ajar.

### API3 — Broken Object Property Level Authorization

The merger of mass assignment and excessive data exposure, and the one this post is most about. The *write* half is the over-posting we worked through: a client sets `status` or `is_admin` because you bound the whole body. The *read* half is excessive data exposure: your response serializes the raw `Order` model, which includes `internal_risk_score`, `customer_id`, and a full `card_pan` (primary account number), trusting the client (often a mobile app) to "only show the public fields." But the API returned them, so they are on the wire and in the attacker's hands.

The symmetric fix is **DTOs on both sides.** An input DTO names the writable fields (no `status`); an output DTO (a response model) names exactly the fields this caller may see (no `card_pan`, only `card_last4`). Never serialize a persistence model directly to the client. This is the same allowlist discipline applied to object *properties* rather than whole objects.

```python
class OrderResponse(BaseModel):
    id: str
    status: str
    total: int
    currency: str
    card_last4: str          # never card_pan
    # internal_risk_score, customer_id, cost_basis: NOT exposed

    @classmethod
    def from_order(cls, o):  # explicit projection, allowlist of fields
        return cls(id=o.id, status=o.status, total=o.total,
                   currency=o.currency, card_last4=o.card_pan[-4:])
```

### API4 — Unrestricted Resource Consumption

A single caller, or a single request, consumes unbounded CPU, memory, bandwidth, or money (every third-party call you make on their behalf — an SMS, an email, a paid API — has a cost). `GET /orders?page_size=1000000` returns a million rows; a 50 MB upload exhausts memory; an unbounded fan-out triggers ten thousand downstream calls. The defenses are the limits from earlier — bounded page sizes, payload caps, depth limits — plus [rate limiting and quotas](/blog/software-development/api-design/rate-limiting-quotas-and-abuse-protection) that bound request frequency, plus spend caps on the costly downstreams. Return `429` with `Retry-After` when a caller exceeds their rate, and `413` when a payload exceeds the cap.

### API5 — Broken Function-Level Authorization

BOLA's sibling for *operations* rather than *objects*. The user may access *some* functions but not the admin ones, and the check is missing or relies on the UI hiding the button. A normal user issues `DELETE /orders/{id}` or `POST /admin/refunds` and it succeeds because the route only checked "are you logged in," not "may you perform this operation." The fix is to authorize the *operation* explicitly — deny by default, and require the specific scope or role (`refunds:write`, `role:admin`) for each privileged function, enforced server-side, never inferred from the client. Again, this lives in the [authorization](/blog/software-development/api-design/authorization-scopes-roles-and-resource-level-permissions) post; the rule is that every function has an authorization check, and "the UI won't show it" is not one.

### API6 — Unrestricted Access to Sensitive Business Flows

New in 2023, and the most interesting because the endpoint is working *as designed* — the abuse is in the *flow*, run at a speed or scale a human never would. A `POST /discounts/claim` is fine for a human claiming one code; a script claiming every code in the catalog in ten seconds drains the promotion. A `POST /orders` is fine; ten thousand of them reserving limited inventory (then never paying) is a scalping attack. Buying out concert tickets, creating fake accounts to farm signup bonuses, scraping your whole catalog — all use legitimate endpoints. The defense is not input validation; it is recognizing which flows are *sensitive* (have business value when automated) and adding flow-level protections: device fingerprinting, proof-of-work or CAPTCHA on the sensitive step, per-flow rate limits keyed to the business action (not just the HTTP route), and anomaly detection on velocity. You are defending the *business invariant* ("one discount per customer"), which no schema can express.

### API7 — Server-Side Request Forgery

The full worked example is above. One-line recap: a URL the client controls becomes a request your server makes, and your server is inside the network. Defense: scheme allowlist, resolve-then-validate the IP against blocked internal ranges, and route outbound client-driven fetches through an egress proxy that cannot reach the inside.

### API8 — Security Misconfiguration

The breach that is not a code bug but a config left wrong. Verbose error responses that leak stack traces, table names, and library versions; CORS set to `Access-Control-Allow-Origin: *` with credentials so any site can read authenticated responses; debug endpoints reachable in production; default credentials; missing the security headers from earlier; TLS misconfigured; an unpatched gateway. A concrete one for us: a `500` on `/payments` returns the full exception including the database connection string. The defense is hardened, reviewed configuration as code — generic error bodies in production (a [problem+json](/blog/software-development/api-design/error-design-a-machine-readable-human-friendly-contract) envelope with a `type`, `title`, and `status`, and *no* internal detail), `nosniff` and `HSTS` on, CORS scoped to known origins, debug off, and the whole config in version control so a review can catch the open door.

```json
{
  "type": "https://api.example.com/problems/internal-error",
  "title": "Internal Server Error",
  "status": 500,
  "instance": "/payments/pay_771",
  "detail": "An unexpected error occurred. Reference: req_a1b2c3."
}
```

That body is safe: it gives the client a correlation id (`req_a1b2c3`) to quote to support, and the actual stack trace stays in your logs where it belongs.

### API9 — Improper Inventory Management

You cannot defend an endpoint you forgot exists. *Shadow APIs* are undocumented endpoints (a debug route, a partner integration nobody tracked); *zombie APIs* are old versions left running after they should have been retired (`/v1/orders` still live, unpatched, unmonitored, after everyone "moved to" `/v2`). Breaches love zombie versions because they have the old, unfixed bugs and nobody is watching them. The defense is *inventory*: maintain a complete catalog of every API, every version, and every environment; know which are deprecated and when they sunset; monitor and patch *all* of them; and actually retire old versions on the schedule you promised. This is the operational payoff of the [deprecation and sunset](/blog/software-development/api-design/deprecation-and-sunset-retiring-an-api-humanely) discipline — a version you sunset is a version that can no longer be the zombie that gets you breached. Spec-first OpenAPI plus a gateway that only routes registered routes turns "what endpoints do we have?" into a query, not an archaeology project.

### API10 — Unsafe Consumption of APIs

The newest entry, and the one that flips the lens: you are not only an API *provider*, you are an API *consumer*, and you tend to trust the APIs you call far more than the clients who call you. A third-party currency-rate API, a partner's order-status webhook, a payment processor's response — if you ingest their data without validation, an attacker who compromises *them* (or a malicious partner) injects into *you*. Concretely: you call a currency-rate service and store its response directly; a poisoned response sets a rate of `0` and now orders price to nothing. Or you follow a redirect the third party returns straight into an internal fetch (their SSRF becomes yours). The defense: treat upstream responses as untrusted input — validate them against a schema exactly as you validate client input, set timeouts and size limits on outbound calls, do not blindly follow redirects from third parties, and isolate third-party integrations so a bad one cannot reach your internals. The boundary works both ways: input is hostile whether it comes from a client or from a service you call.

## Putting it together: one validation middleware

The controls above are not eight separate libraries you bolt on; they compose into one boundary layer that runs on every request before any handler. Building it as middleware (a function the framework runs around your handlers) means a new endpoint inherits the floor for free, which is the only way the floor actually holds across a team — the moment validation is a thing each handler must remember to call, some handler forgets. Here is the spine of such a layer for our Payments API, in the order the gates fire.

```python
async def secure_boundary(request, call_next):
    # 1. SIZE: reject oversized bodies before parsing (cheap, kills floods)
    if request.headers.get("content-length", "0").isdigit() and \
       int(request.headers["content-length"]) > MAX_BODY_BYTES:
        return problem(413, "payload-too-large", "Body exceeds 256 KB.")
    # 2. CONTENT-TYPE: only accept what we can safely parse
    if request.method in ("POST", "PUT", "PATCH") and \
       request.headers.get("content-type", "").split(";")[0] != "application/json":
        return problem(415, "unsupported-media-type", "Send application/json.")
    # 3. AUTHN handled upstream; identity is on request.state.principal here
    response = await call_next(request)      # handler validates body via its DTO
    # 4. OUTPUT hardening on the way out, on every response
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Content-Type"] = "application/json; charset=utf-8"
    response.headers.setdefault("Cache-Control", "no-store")
    return response

def problem(status, code, detail):
    body = {"type": f"https://api.example.com/problems/{code}",
            "title": code.replace("-", " ").title(), "status": status,
            "detail": detail}
    return JSONResponse(body, status_code=status,
                        media_type="application/problem+json")
```

Notice what is *not* in the middleware: object-level and function-level authorization. Those cannot live here, because the middleware does not know that `ord_8812` belongs to this caller — only the handler, with the data loaded, knows that. The middleware does the generic, context-free work (size, content type, response hardening, the problem-envelope helper) and hands a clean, well-typed request to a handler that does the context-aware work (DTO binding, ownership check, business rule, parameterized store). That division — generic at the edge, specific in the handler — is the architecture the next section makes explicit. And every rejection returns a `problem+json` body, so a client gets a machine-readable, leak-free error rather than a stack trace, which closes the misconfiguration door (API8) on the same code path.

**Stress test the boundary.** What happens when a client sends `Content-Type: application/json` but the body is not JSON? The deserializer throws; the DTO binding fails; the handler's framework returns a `422` from the typed validator — fail closed, no handler logic runs on garbage. What happens when the body is valid JSON but 256 KB plus one byte? The size gate at step 1 returns `413` before the parser ever sees it. What happens when validation itself is slow under load? Steps 1 and 2 are string comparisons with no I/O, so they cannot be the bottleneck; the expensive step is the handler's database work, which is gated behind authentication and the size/type checks, so a flood of unauthenticated oversized junk dies at constant cost. What happens when an endpoint is added without a DTO? It accepts `dict` and the boundary cannot protect it — which is why the governance answer is a lint rule (a Spectral or custom check) that fails CI when a handler binds an untyped body, turning "remember to use a DTO" into a gate the machine enforces rather than a habit the human must keep.

## Where each control lives

Security controls are not a pile; they are a sequence, and putting each one in the right place is half the battle. A control too late is a control some path skipped; a control in the wrong layer is a control that does not have the information it needs (the gateway cannot do object-level authorization because it does not know who owns the order; only the service does).

![a timeline showing controls from TLS and gateway through authenticate validate authorize encode output and the parameterized store layer](/imgs/blogs/input-validation-output-encoding-and-the-owasp-api-top-10-8.png)

| Control | Lives at | When to use it there | When NOT to |
| --- | --- | --- | --- |
| TLS, size caps, header hygiene | Gateway / edge | Always; cheapest rejections, shared policy | Never skip; do not rely on it for app logic |
| Authentication (who) | Gateway or service edge | Verify the credential before any handler runs | Don't infer identity from a client-set header |
| Schema validation, DTO bind (what) | Service boundary | The instant JSON becomes typed values | Don't validate deep in the call graph |
| Object/function authorization | Service, in the handler | Where the data and ownership are known | Don't push to the gateway; it lacks context |
| Business-flow protection | Service + anomaly layer | For flows with value when automated | Don't rely on rate limits alone for these |
| Output encoding (contextual) | The consumer, at render | HTML/attr/JS encoding for the render context | Don't pre-escape in the API's JSON body |
| Content-Type, nosniff, CORS | Service / gateway response | On every response | Don't return `*` CORS with credentials |
| Parameterization | The data-access layer (the sink) | Every query, every command, always | Never concatenate input into a query, ever |

The shape to internalize: **edge does the cheap, generic, context-free checks** (TLS, size, content type, authentication, rate limits); **the service does the expensive, specific, context-aware checks** (schema/DTO validation, object and function authorization, business invariants, parameterized access). Output encoding is special — the contextual part is genuinely the consumer's job, and your job is to be honest about the media type and to store the truth.

## Case studies: how these classes actually breach

These are real, well-documented *classes* of API failure, drawn from the public security record and OWASP's own cataloging. The specifics below are general and accurate to the class; I am not attributing invented details to any named company.

**BOLA is the dominant breach class, repeatedly.** Across the public record of API breaches and bug-bounty disclosures, the single most common serious finding is an authenticated user reading or modifying another user's objects by manipulating an identifier — order IDs, account IDs, document IDs in a URL. OWASP elevated it to API1 in both the 2019 and 2023 lists precisely because the breach reports kept showing the same shape: solid authentication, missing object-level authorization. The lesson the industry learned the hard way is that *being logged in is not permission to touch this specific object.*

**Mass assignment has a long, named history.** The most famous early demonstration was against a major code-hosting platform's framework defaults, where binding the whole request body let a researcher add their own key to a project they did not own — a textbook over-post. It is why "strong parameters" became a framework default in that ecosystem, and why every mature framework now ships an allowlist-binding mechanism. The 2023 merger into API3 reflects that the same root cause — the API trusting the client to respect property boundaries — covers both setting fields you should not and seeing fields you should not.

**SSRF and cloud metadata.** A widely-analyzed class of cloud breach combined an SSRF in an application (often via a URL-fetch or webhook feature, sometimes via a misconfigured proxy) with the cloud instance metadata service handing out credentials to whatever asked from inside. The pattern is general and well documented in cloud-security guidance: an externally-reachable request-forgery primitive plus a metadata endpoint that trusts network position equals credential theft. It is the reason cloud providers shipped hardened metadata services that require a deliberate token step, and the reason "block link-local in your fetcher" is now standard.

**Improper inventory and the zombie version.** Several high-profile API exposures traced back not to the current, well-guarded API but to an old version or a forgotten staging endpoint that was still reachable, still held data, and was no longer monitored or patched. The 2023 list's emphasis on inventory is a direct response: the endpoint you forgot is the endpoint that has your oldest, unpatched bug. The OWASP guidance and the broader API-security literature converge on the same operational rule — you must have a complete, current inventory of every API and version, and you must actually retire the old ones.

The thread through all four: these are not exotic exploits. They are an API trusting input, or an authorization check that is missing, or a config left at an insecure default, or an endpoint nobody remembered. The OWASP API Top 10 reads, top to bottom, as a list of *design and authorization* failures — which is exactly why it is in scope for an API-design series rather than only a pen-testing one.

## When to reach for each control (and when not to)

Security advice that is all "always do everything" is useless, because you ship nothing. Here is the decisive version.

**Always, no exceptions:** parameterize every query (there is no case where concatenating input into SQL is correct); validate request bodies against a schema or typed DTO at the boundary; bind only an allowlist of writable fields; project responses through an allowlist of viewable fields; set `Content-Type` and `X-Content-Type-Options: nosniff` on every response; cap payload size; authenticate before any handler runs; check object ownership on every object access. These are cheap and the failure mode of skipping them is a breach. There is no trade-off to weigh.

**Validate at the boundary, but do not double-validate everywhere.** Once a value is a typed, validated DTO, the rest of your code can trust it; re-validating the same `quantity` in five layers is noise that hides the one layer that matters. Validate once, at the edge, thoroughly, and let the type system carry the guarantee inward.

**Reach for an egress proxy / network isolation for SSRF when you have *any* client-driven outbound fetch** (webhooks, URL imports, image fetch). For a service with no such feature, the IP-block validator is enough; do not build an egress gateway you do not need.

**Reach for business-flow protections (CAPTCHA, proof-of-work, velocity detection) only for flows that have value when automated** — discount claims, signups with bonuses, limited-inventory purchases, bulk lookups. Do not put a CAPTCHA on `GET /orders/{id}`; you punish real users to defend a flow with no automation value, and you have not stopped a determined attacker anyway.

**Do not return `200` with an error body, ever.** Use the status code (`400`/`401`/`403`/`404`/`409`/`422`/`429`/`500`); a `200` with `{"error": ...}` defeats every client, cache, and monitor that reads status, and it is its own kind of misconfiguration.

**Do not rely on the client for security.** Hiding a button, obfuscating a field name, "the mobile app filters it out," client-side validation as the *only* validation — every one of these is bypassed by anyone who reads the wire. Client-side checks are for UX (fast feedback); server-side checks are for security. Both, never only the first.

**Do not trust input because of where it came from.** A request from your own mobile app, from a logged-in user, from a paying partner, or from a third-party API you call — all of it is input, all of it is hostile until validated. The most damaging breaches come from over-trusting an authenticated insider, which is why BOLA and API10 are on the list at all.

## Key takeaways

- **Validate at the boundary, deny by default, fail closed.** A value is validated at the edge or it does not enter the system; the default decision is reject; the error path denies.
- **Allowlist, never denylist.** An allowlist fails closed on inputs you never anticipated; a denylist fails open on the one rule you forgot. Pay in support tickets, not in incidents.
- **Bind only the fields the client may write.** Never bind a request body to a persistence model — use a DTO whose fields are exactly the writable ones, so over-posting `status` or `is_admin` does nothing. Project responses the same way so you never leak `card_pan`.
- **Parameterize every query.** Injection is crossing the data-to-code boundary; keep the boundary fixed by binding values as parameters. Validation rejects garbage, but only parameterization makes injection structurally impossible.
- **Even a JSON API feeds XSS.** Set `Content-Type` and `X-Content-Type-Options: nosniff`, store the truth, and let each consumer encode for its own render context.
- **Treat every client-supplied URL as an SSRF.** Allowlist the scheme, resolve and validate the IP against internal ranges, and route client-driven fetches through an egress path that cannot reach the inside.
- **Bound every input dimension.** Bytes, depth, breadth, count — each gets a maximum, enforced cheaply at the door, because an unbounded input is a denial of service.
- **Most of the OWASP API Top 10 is authorization and design.** API1/API3/API5 are authorization failures; API6 is business-logic abuse; API8/API9 are config and inventory. Fix them in your design and your authorization layer, not with a scanner.
- **Canonicalize before you validate, and validate the form you will actually use.** Decode once, normalize once, validate that, use that — so two encodings can never sneak past one check.

This is the defensive-coding floor for any API you ship. With authentication answering *who*, authorization answering *what may you do*, rate limiting answering *how much*, and this post answering *can I trust what you sent*, the security track is complete — and it all rolls up into the [API design playbook](/blog/software-development/api-design/the-api-design-playbook-a-review-checklist-first-endpoint-to-v2), where "validate input, bind a DTO, parameterize, authorize the object, encode output, set nosniff" becomes a line on the review checklist you run before every endpoint ships.

## Further reading

- **OWASP API Security Top 10 (2023)** — the canonical list, with each risk's description, example, and prevention. The source for everything in the walkthrough above.
- **OWASP Cheat Sheet Series** — the Input Validation, Mass Assignment, SQL Injection Prevention, SSRF Prevention, and REST Security cheat sheets; practical, control-by-control guidance.
- **JSON Schema (draft 2020-12)** — the specification for declaring and validating request shapes, including `additionalProperties`, `enum`, `pattern`, and the bounds keywords.
- **RFC 9457 — Problem Details for HTTP APIs** — the `problem+json` error envelope used in the misconfiguration example; how to return errors without leaking internals.
- **OpenAPI Specification 3.1** — spec-first request validation and the basis for runtime schema enforcement and an accurate API inventory.
- **OWASP ASVS (Application Security Verification Standard)** — a checklist-style standard you can map your validation, authorization, and output-handling controls against for a security review.
- Within this series: the security siblings — [authentication](/blog/software-development/api-design/authentication-api-keys-sessions-jwt-and-mtls), [authorization](/blog/software-development/api-design/authorization-scopes-roles-and-resource-level-permissions), [rate limiting](/blog/software-development/api-design/rate-limiting-quotas-and-abuse-protection), and the [filtering and sorting](/blog/software-development/api-design/filtering-sorting-and-sparse-fieldsets-without-reinventing-sql) post for safe query expressiveness — plus the [intro hub](/blog/software-development/api-design/what-is-an-api-the-contract-between-systems) and the [capstone playbook](/blog/software-development/api-design/the-api-design-playbook-a-review-checklist-first-endpoint-to-v2).
