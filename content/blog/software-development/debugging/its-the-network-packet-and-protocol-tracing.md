---
title: "It's the Network: Packet and Protocol Tracing for the Skeptical Engineer"
date: "2026-06-20"
publishDate: "2026-06-20"
description: "Prove what actually crossed the wire with tcpdump, tshark, and ss — turn it's never the network into a packet you can point at."
tags:
  [
    "debugging",
    "software-engineering",
    "networking",
    "tcpdump",
    "wireshark",
    "tcp",
    "tls",
    "packet-capture",
    "protocol-tracing",
  ]
category: "software-development"
subcategory: "Debugging"
author: "Hiep Tran"
featured: true
readTime: 55
image: "/imgs/blogs/its-the-network-packet-and-protocol-tracing-1.png"
---

There is a sentence every infrastructure engineer has said and every developer has rolled their eyes at: "It's never the network." It is a good prior. Most of the time the network is fine and your code, your config, or your assumptions are the problem. But the prior is dangerous precisely because it is usually right, because it trains you to stop looking at the one place where the truth is actually sitting, in plaintext, waiting for you. The wire does not lie. The wire does not have opinions. The wire has bytes, and exactly the bytes that were sent, in exactly the order they arrived, with timestamps. If you can read them, you can end an argument that three teams have been having for two hours.

Here is the scene that makes people believe in packet capture. A service starts returning errors. The client team says "the API is down." The API team pulls up their dashboards and says "we're serving fine, our error rate is normal, it's your client or the network." Both are looking at their own logs, which is to say both are looking at the half of the story that flatters them. Nobody has looked at the cable. Then somebody runs `tcpdump` on the client host for ninety seconds, opens the capture, and sees a SYN packet leave for the server's port 443 and an RST come straight back four-tenths of a millisecond later. That single fact ends the meeting. The SYN left — so it is not a client networking problem and not DNS. The RST came back instantly — so the host is reachable but nothing is listening on that port, or a firewall is actively refusing. The "API is down" was half right and the "it's your network" was wrong, and neither team could have known which until somebody read the wire.

![A vertical stack showing the six layers a single HTTPS request crosses, from the application call through DNS resolution, the TCP SYN, the TLS ClientHello, and the HTTP bytes, ending in a danger box noting any hop can drop the request before server code runs](/imgs/blogs/its-the-network-packet-and-protocol-tracing-1.png)

This post is the field manual for that move. By the end you will be able to capture exactly the traffic you care about with `tcpdump`, read it with Wireshark and `tshark`, follow a single TCP stream, recognize a three-way handshake, tell an `RST` from a `FIN`, spot retransmissions and zero-windows, prove that a request never left the box, read a TLS handshake well enough to name a cert or SNI failure, catch DNS resolving to a stale IP, and diff what went into a proxy against what came out the other side. We will stay on the series' spine the whole way: **observe → reproduce → hypothesize → bisect → fix → prevent**. Packet capture is the highest-resolution `observe` step you own, and it is the one that turns "the network is acting up" — an un-falsifiable feeling — into "here is the SYN, here is the RST, the port is closed" — a hypothesis you can confirm in one command. If you have not read [the scientific method of debugging](/blog/software-development/debugging/stop-guessing-the-scientific-method-of-debugging), start there; this post is that method applied to the one layer everyone is afraid to open.

## 1. Why "it's never the network" is a trap, and what the wire actually proves

Let us be precise about what we are even claiming when we say "the network." A modern HTTP or RPC call is not one thing that either works or does not. It is a sequence of independent steps, each of which can fail on its own, each of which produces a different and recognizable signature on the wire. The application calls a client library. The library resolves a name to an IP via DNS. The kernel opens a TCP connection with a three-way handshake. If it is HTTPS, a TLS handshake negotiates encryption on top of that connection. Only then do the actual request bytes go out, and only then does any server-side code run. A failure at any of the first four steps will look, from the application's point of view, like "the server didn't answer" — even though the server never saw a single byte.

This is why the blame argument is so persistent and so useless. The client's stack trace says `connection timed out` or `connection refused` or `handshake failed`, and the application developer reads that as "the other side broke." The server's logs say nothing at all, because the request never arrived, and the server developer reads *that* as "we're healthy, not our problem." Both readings are reasonable and both are guesses. The only way to know which step failed is to look at the bytes that did or did not cross the wire, and that is exactly what a packet capture gives you: a timestamped, ordered record of every frame that hit the network interface, independent of what any application chose to log about it.

The mechanism worth internalizing is that **the network stack is layered, and a failure low in the stack masquerades as a failure high in the stack.** When DNS returns a stale IP, the TCP SYN goes to a machine that has been decommissioned, the SYN gets no answer, the connection times out after the OS retry budget is exhausted, and your HTTP client raises a timeout. The application sees an HTTP-level symptom. The actual fault is two layers down and several seconds earlier, in a cached A record. No amount of staring at the HTTP client's error message will reveal that, because the HTTP client never got far enough to have an opinion. The capture, by contrast, shows the SYN going to an IP you do not recognize, and the mystery evaporates.

So the first discipline is to stop asking "is it the network?" as a yes-or-no question and start asking "which layer, and what's the signature?" The figure above lays out the layers a single request passes through. Each one is a place a call can die silently. The rest of this post is a tour of those layers with the exact command that exposes each one and the exact packet pattern that names the fault. The wire does not prove the network is to blame. The wire proves *which step actually happened and which did not*, and that is far more useful, because half the time it exonerates the network and points the finger squarely back at your own DNS cache, your own firewall rule, or your own code that opened a socket and forgot to close it.

| Symptom the app reports | What it feels like | What only the wire can tell you |
| --- | --- | --- |
| `connection refused` | server is down | SYN left, RST came back instantly — port closed, host is up |
| `connection timed out` | server is slow/down | SYN left, nothing came back — firewall dropping, or wrong IP |
| `handshake failed` | TLS is broken | ClientHello sent, no ServerHello — cert/SNI/version mismatch |
| `read timeout` | server hung | connection established, request sent, no response bytes — server stuck |
| intermittent 5xx | flaky service | retransmissions and a 200ms RTO — the link is lossy, not the service |
| works locally, fails in prod | "weird" | different DNS answer, an extra proxy hop, a smaller MTU |

Keep that table near you. Every row is a different bug, and the application-level symptom in the left column is nearly useless for telling them apart — the right column, the thing only a capture shows you, is what closes the case.

## 2. tcpdump: capturing exactly the traffic you want and nothing else

`tcpdump` is the workhorse. It is on essentially every Linux box, it has no GUI, it writes a standard `.pcap` file that every other tool can read, and it is built around a small filter language that lets you grab precisely the conversation you care about out of a firehose. The single most common mistake people make with it is capturing everything, which on a busy host means gigabytes of noise you will never wade through, or capturing on the wrong interface, which means capturing nothing at all. Both are avoidable.

Here is the invocation you will reach for ninety percent of the time:

```bash
# Capture all traffic to/from host 203.0.113.10 on port 443,
# on every interface, write it to a file for later analysis.
sudo tcpdump -i any -w cap.pcap host 203.0.113.10 and port 443

# ...reproduce the bug while this runs, then Ctrl-C.

# Read it back, with names un-resolved (-n) so you see real IPs/ports,
# and verbose enough to see flags and sequence numbers.
tcpdump -n -r cap.pcap
```

Three flags carry the weight. `-i any` captures on all interfaces, which saves you from the classic failure of capturing on `eth0` when the traffic actually went out `eth1` or over a tunnel. (Once you know the interface, name it explicitly — `-i eth0` — because `any` on some kernels captures in a cooked mode that hides link-layer details.) `-w cap.pcap` writes raw packets to a file instead of printing summaries; **always capture to a file** for anything non-trivial, because the live decode is lossy and you will want to re-read the same capture through several tools and several filters. And `-n` on read disables name resolution so you see `203.0.113.10.443` instead of a hostname, which matters because reverse-DNS lookups during a capture both slow you down and add their own traffic to the very thing you are trying to observe.

The filter — `host 203.0.113.10 and port 443` — is a BPF (Berkeley Packet Filter) expression, and it is worth a few minutes of your life to learn because it is the difference between a 30 KB capture you can read and a 3 GB capture you cannot. BPF runs in the kernel, so filtered-out packets are dropped before they ever reach `tcpdump`, which keeps overhead low even on a loaded box. A few you will use constantly:

```bash
# Just one host, one port:
host 203.0.113.10 and port 443

# A whole subnet:
net 10.2.0.0/16

# Only the connection setup/teardown packets (SYN, FIN, RST) —
# great for "did the handshake even start?" with almost no volume:
'tcp[tcpflags] & (tcp-syn|tcp-fin|tcp-rst) != 0'

# DNS traffic only:
udp port 53

# Exclude your SSH session so you don't capture your own keystrokes:
not port 22
```

That third filter — capturing only packets with a SYN, FIN, or RST flag set — is a small piece of magic for the "is anything even happening?" question. On a host doing thousands of requests a second, it reduces a flood to just the handshake and teardown events, which is exactly what you need to answer "did a connection open, and how did it close?" without drowning. The `tcp[tcpflags]` syntax reaches into the TCP header's flags byte; the named constants (`tcp-syn`, `tcp-fin`, `tcp-rst`, `tcp-ack`, `tcp-push`) make it readable.

Two operational notes that save real grief. First, on a high-traffic interface, add `-c 2000` to stop after 2000 packets or `-G 60 -W 1` to rotate after 60 seconds, so you do not fill the disk while you go get coffee. Second, `tcpdump` truncates each packet to a default snap length on older versions; modern versions capture the full packet, but if you are on something ancient and your payloads look cut off, add `-s 0` to capture full frames. The goal is always the same: the smallest capture that contains the bug, captured on the interface the bug actually used.

#### Worked example: proving "the API is down" is the wrong diagnosis

A payments service started throwing `connection refused` to a downstream risk-scoring API. The on-call engineer's first instinct, and everyone's, was that the risk API had crashed. Its owners disagreed; their pods were healthy and serving traffic from other clients. Two hours of "it's you / no it's you" later, somebody captured on the payments host:

```bash
sudo tcpdump -i any -w refused.pcap host 10.4.12.7 and port 8443
# trigger one payment, Ctrl-C, then:
tcpdump -n -r refused.pcap
```

The output was two lines and they ended the argument:

```bash
# tcpdump -n -r refused.pcap  (representative of what it prints):
12:04:01.118233 IP 10.4.30.51.51442 > 10.4.12.7.8443: Flags [S], seq 281
12:04:01.118661 IP 10.4.12.7.8443 > 10.4.30.51.51442: Flags [R.], seq 0, ack 282
```

The first line is a SYN (`Flags [S]`) leaving the payments host for `10.4.12.7` port `8443`. So the request *did* leave the machine — it is not a DNS problem, not a local firewall on the client, not the client library. The second line, 0.4 milliseconds later, is an `R.` — an RST with the ACK bit — coming straight back from the server's IP. The host is up and answering. But it answered with a reset, which means *nothing was listening on port 8443*. The risk API had been redeployed an hour earlier and was now listening on `8444`; the payments service's config still pointed at `8443`. Two minutes of capture beat two hours of dashboards, because the dashboards each showed one side of the conversation and the capture showed the conversation itself. The fix was a one-line config change. The lesson was that "the API is down" and "it's your network" were both confident, both wrong, and both un-checkable until somebody read the wire. The IPs and timings here are representative of what these tools print, sized to be believable rather than copied from one specific incident; when you run it yourself the format is exactly this.

## 3. The three-way handshake, RST vs FIN, and reading a connection's life

You cannot debug TCP without a working picture of how a connection is born and how it dies, because half of all network bugs are visible purely in the setup and teardown packets, before any payload is even involved. The good news is that the picture is small. A TCP connection opens with three packets — the famous three-way handshake — and closes one of exactly two ways, and the difference between those two ways is one of the most useful single facts in network debugging.

The handshake: the client sends a `SYN` (synchronize) packet, announcing it wants to talk and carrying its initial sequence number. The server replies with a `SYN-ACK`, acknowledging the client's SYN and sending its own. The client sends a final `ACK`. Three packets, and now the connection is `ESTABLISHED` and data can flow. In a `tcpdump` read you will see exactly this:

```bash
# tcpdump read of a healthy handshake (three packets):
IP client.51442 > server.443: Flags [S],  seq 1000
IP server.443 > client.51442: Flags [S.], seq 9000, ack 1001
IP client.51442 > server.443: Flags [.],  ack 9001
```

`[S]` is SYN, `[S.]` is SYN-ACK (the dot is the ACK flag), `[.]` is a bare ACK. If you see all three, you have a connection. If you see the first `[S]` repeated several times with no `[S.]` ever coming back, the SYN is being dropped — a firewall is silently eating it, or you are talking to the wrong port, or the host is unreachable. The *absence* of the SYN-ACK is the entire diagnosis, and you would never see it from the application side, which only knows that eventually it timed out.

![A branching graph showing a TCP connection opening with SYN, SYN-ACK, and ACK, then data flowing, then splitting into two teardown outcomes: a clean FIN-and-ACK close marked success and an RST marked refused or abort](/imgs/blogs/its-the-network-packet-and-protocol-tracing-2.png)

Now the teardown, and this is the high-value part. A connection can close cleanly or violently, and the wire shows which. A **clean close** is a `FIN` (finish): one side says "I'm done sending," sends a `FIN`, the other acknowledges, and after both directions have FINed the connection winds down through `TIME_WAIT`. A FIN means "I have finished, in an orderly way." A **violent close** is an `RST` (reset): one side says "this connection is over, right now, discard everything." An RST means "I refuse" or "something is wrong, abort." The figure above shows both exits from the same established connection.

That distinction resolves an enormous number of bugs. An RST arriving immediately after your SYN means the port is not listening — connection refused. An RST arriving in the middle of an established connection means something abruptly killed it: the server crashed, a load balancer's idle timeout fired, a firewall decided the connection was stale and reset it, or the application called `close()` on a socket that still had unread data and the kernel sent an RST instead of a graceful FIN. A FIN in the middle of what you expected to be a long-lived connection means the other side closed it *gracefully* but sooner than you wanted — often a keep-alive timeout you did not know about. "We get random connection drops" becomes a precise question once you know whether those drops are FINs (somebody is closing on purpose, find the timeout) or RSTs (something is aborting, find the killer).

There is a subtlety worth flagging because it bites people. An application that does `close()` while data is still sitting unread in its receive buffer causes the kernel to send an RST rather than a FIN, because there is no clean way to finish a conversation you are abandoning mid-sentence. So a flood of RSTs is not always a network villain; sometimes it is your own service closing connections abruptly under load, perhaps because a request handler threw and the framework yanked the socket. The capture tells you the RST happened and who sent it; *why* that side sent it is the next hypothesis to chase, often back in your own code. This is the loop in miniature: the packet is the observation, "who reset and why" is the next falsifiable question.

## 4. Wireshark and tshark: following a stream and reading the conversation

`tcpdump` captures and gives you a terse per-packet summary. To actually *read* a conversation — to follow a single TCP stream from handshake to teardown, reassemble the HTTP request and response, and see retransmissions highlighted — you want Wireshark (the GUI) or `tshark` (its command-line sibling, perfect for servers where you cannot run a GUI). The crucial workflow they unlock is **Follow TCP Stream**: pick one packet, and the tool isolates every packet belonging to that one connection and reassembles the byte stream in each direction, so you see the actual request and response as the application saw them, not as 47 scattered frames.

In the Wireshark GUI you right-click a packet and choose Follow → TCP Stream. From the command line, `tshark` does the same with filters:

```bash
# List the TCP conversations in a capture, so you can pick the one you want:
tshark -r cap.pcap -q -z conv,tcp

# Follow a specific stream by its index (stream 3 here),
# printing the reassembled bytes:
tshark -r cap.pcap -q -z follow,tcp,ascii,3

# Or filter to one connection by the 4-tuple and show HTTP:
tshark -r cap.pcap -Y 'tcp.stream eq 3 and http' -V
```

The `-z conv,tcp` gives you a table of every connection in the capture with packet and byte counts in each direction — instantly useful, because a connection with bytes going out but zero coming back is a stuck request, and a connection with a wildly lopsided retransmit count is a lossy one. The `follow,tcp` reassembles a stream into readable text. And `-Y` applies a *display filter* (Wireshark's richer filter language, distinct from the BPF capture filter) so you can say things like `http.response.code == 504` or `tcp.analysis.retransmission` or `tls.handshake.type == 1` and see only the packets that match.

Display filters are where Wireshark earns its reputation, because Wireshark has dissectors for hundreds of protocols and will compute things for you that are not literally in any single packet. The ones I reach for constantly:

```bash
tcp.flags.reset == 1                  # every RST in the capture
tcp.analysis.retransmission           # packets Wireshark flagged as resent
tcp.analysis.zero_window              # receiver advertised a full buffer
tcp.analysis.duplicate_ack            # dup-ACKs (a loss signal)
http.response.code >= 500             # server errors at the HTTP layer
tls.handshake.type == 1               # ClientHello packets
dns.flags.rcode != 0                  # DNS responses that returned an error
```

`tcp.analysis.retransmission` deserves a word because it is not a flag in the packet — Wireshark *derives* it by noticing that a sequence number it already saw is being sent again. That is exactly the kind of cross-packet reasoning you would do by hand, automated. When you filter a capture to `tcp.analysis.retransmission` and the result is a long list, you have found a lossy link without measuring anything yourself; Wireshark counted the resends for you. Pair it with the "Statistics → Conversations" view, which shows per-connection round-trip time and retransmit rate, and you can rank connections by how sick they are.

The mental shift here is from "packets" to "conversations." A capture is a pile of frames, but the bug lives in a conversation: one client talking to one server, one stream that did or did not complete. Follow TCP Stream collapses the pile into the story. When someone hands you a 200,000-packet capture and says "the checkout is slow," you do not read 200,000 packets — you find the checkout connection (filter by the client IP and the server port), follow that one stream, and read its life from SYN to whatever went wrong, which is usually one of the six signatures from the table in section 1.

## 5. The request that never left: SYN, no SYN, and proving it is local

The most clarifying capture result is the *absence* of a packet. When an application says it cannot reach a server, there is a binary fact that splits the universe of causes in half: did a SYN leave this machine or not? If a SYN left and you can see it on the wire, the problem is somewhere between you and the server — routing, firewall, the server's port, the server itself. If **no SYN left at all**, the problem is entirely local, before any packet was ever generated: DNS failed to resolve, a local firewall (`iptables`, a security group's egress rule, a `NetworkPolicy` in Kubernetes) blocked the connection before it hit the wire, the connection pool was exhausted so the client never even tried, or the application was pointed at the wrong address and is "connecting" to nowhere.

This is the single most powerful split in network debugging, so let us make it operational. Capture, then reproduce, then check whether any SYN to the target appears:

```bash
# Capture only SYNs to the target host (tiny capture, instant to read):
sudo tcpdump -i any -n 'tcp[tcpflags] & tcp-syn != 0 and host 203.0.113.10'

# In another terminal, trigger the failing request.
# Then look: is there a SYN in the output, or nothing?
```

If you see a line — `IP 10.0.0.5.x > 203.0.113.10.443: Flags [S]` — the SYN left, and you have eliminated the entire class of local problems in one shot. If you see *nothing* while the application is actively erroring, the request never reached the network, and you now know to look locally and not waste another second blaming the far side. That negative result is worth more than most positive ones, because it redirects an entire investigation.

When the SYN never leaves, walk these in order, fastest first:

```bash
# 1. Does the name even resolve, and to what?
dig +short api.internal.example.com
# (compare to what you expect; a stale or empty answer explains everything)

# 2. Is something blocking egress locally?
sudo iptables -L OUTPUT -n -v        # legacy firewall counters
# In k8s, check NetworkPolicy and the CNI; in cloud, security-group egress.

# 3. Is the connection pool exhausted (so the app never dials)?
ss -s                                 # socket summary; look for the totals
ss -tan | grep -c SYN-SENT            # connections stuck dialing
```

The `dig` check is first because DNS is the most common "request never left in a useful direction" cause and it takes one second to rule in or out. We will give DNS its own section because the failure mode is subtle — resolving to a *wrong* IP is worse than resolving to nothing, since the SYN does leave, just toward a ghost. But the gross case, where the name does not resolve at all, shows up as no SYN and a clear `dig` failure, and you are done.

#### Worked example: the firewall rule that ate the SYN

A batch job that had run nightly for a year suddenly could not reach an internal reporting database. The error was `connection timed out` after 30 seconds — the classic "server is slow or down" symptom. The DBA swore the database was up, and it was; other clients connected fine. The capture on the batch host told the story by what it did *not* contain:

```bash
sudo tcpdump -i any -n 'host 10.6.0.40 and port 5432'
# run the job... and the capture stayed completely empty.
```

Not a single packet. No SYN, nothing. The job was not timing out because the database was slow; it was timing out because *no packet ever left the box*. That immediately ruled out the database, the route, and the far-side firewall. A local `iptables -L OUTPUT -n -v` showed a new `DROP` rule with a climbing packet counter on egress to the database subnet — a security hardening change had shipped that afternoon and accidentally blocked the batch host's egress range. The 30-second "timeout" was the kernel's own connect retry budget expiring against a connection that was being dropped before it left. The fix was an allow rule; the diagnosis was the empty capture. An empty capture, when you *expected* packets, is not a failed experiment — it is the answer.

## 6. DNS: resolving to the wrong, stale, or split-horizon IP

DNS is where "the SYN left but went nowhere useful" lives. The dangerous DNS failure is not the name failing to resolve — that errors loudly and is easy. The dangerous one is the name resolving to an IP that is *wrong*: a decommissioned host, a stale record from before a failover, a different answer than you get from your laptop because of split-horizon DNS or a different resolver in the pod. The SYN leaves confidently toward an address that no longer serves you, gets no answer (or gets an RST from whatever now owns that IP), and the application reports a timeout or a refusal that has nothing to do with the service you think you are calling.

The instrument is `dig`, and the discipline is to ask *the same resolver the application uses* and to compare against the truth. `dig` shows you not just the answer but the TTL, which tells you how long a wrong answer will stay cached and keep hurting you:

```bash
# What does this name resolve to right now, and what's the TTL?
dig api.internal.example.com A

# Ask a specific resolver (e.g. the cluster DNS) to match what the pod sees:
dig @10.96.0.10 api.internal.example.com A

# Trace the full delegation from the root, to catch a wrong authoritative answer:
dig +trace api.internal.example.com

# What's actually in the answer, terse:
dig +short api.internal.example.com
```

The output's `ANSWER SECTION` gives you the IP and the TTL in seconds. A common trap: you `dig` from your laptop, get the right IP, and conclude DNS is fine — but the failing process is in a Kubernetes pod using the cluster's DNS, or behind a corporate resolver, or reading a stale `/etc/hosts` entry, and *it* gets a different answer. Always ask the resolver the broken thing uses. Inside a container, `cat /etc/resolv.conf` to see which resolver, then `dig @that-resolver`.

The mechanism behind stale-DNS pain is caching at every level: the authoritative TTL, the recursive resolver's cache, the OS resolver cache (`nscd`, `systemd-resolved`), and the application's *own* cache — many runtimes and HTTP clients cache resolved IPs for the life of the process, sometimes ignoring TTL entirely. So a service can keep hammering a dead IP for hours after DNS was fixed, because the dead IP is pinned in the process's memory. This is why "we updated DNS and it's still broken" is so common: the new record is correct everywhere except inside the long-running process that cached the old one at startup. The capture confirms it — you see SYNs going to the *old* IP while `dig` returns the *new* one, and the gap between those two facts is your bug. The fix is to bounce the process (clearing its cache) or to configure the client to honor TTLs; the diagnosis is the capture-versus-`dig` mismatch.

To tie DNS into the wire-level view, capture the DNS exchange itself and read it:

```bash
# Capture and read the DNS query and response:
sudo tcpdump -i any -n -w dns.pcap udp port 53
tshark -r dns.pcap -Y dns -T fields -e dns.qry.name -e dns.a -e dns.resp.ttl
```

That prints the query name, the A-record answers, and their TTLs straight from the wire, so you see exactly what the resolver told the application — which can differ from what `dig` told you a second later if the cache changed underneath you. When the answer on the wire is the stale IP, you have caught the cache red-handed.

![A two-column before-and-after figure contrasting a guess that the API is down, with no proof anything left the machine, against the proof from a packet capture showing a SYN reached the host and an RST returned in under a millisecond, pointing the fix at the firewall or port](/imgs/blogs/its-the-network-packet-and-protocol-tracing-3.png)

## 7. The signature table: reading the symptom straight off the wire

By now you have seen several distinct on-the-wire signatures, and it is worth collecting them, because once you internalize the catalog, reading a capture becomes pattern matching rather than archaeology. You glance at the first few packets of a connection and the *shape* tells you the layer. The figure below is the lookup table I keep in my head; the prose after it walks each row.

![A matrix mapping six network symptoms to what the packet capture shows and the likely root cause, covering no SYN at all, SYN with no SYN-ACK, SYN then RST, ClientHello with no ServerHello, many retransmits, and a TCP zero window](/imgs/blogs/its-the-network-packet-and-protocol-tracing-4.png)

**No SYN at all.** Nothing leaves the interface for the target while the app errors. The fault is entirely local: DNS, a local/egress firewall, a NetworkPolicy, or an exhausted connection pool. Confirm with `dig` and `iptables -L OUTPUT -v`. This is section 5's empty capture.

**SYN, no SYN-ACK.** The SYN leaves, repeats a few times (the kernel retries — typically at 1s, 2s, 4s intervals), and no SYN-ACK ever returns. The packet is being dropped silently between you and the listener: a firewall that DROPs rather than REJECTs (a REJECT would send an RST; a DROP sends nothing, which is why it manifests as a timeout), a wrong port with nothing listening *and* a firewall swallowing the would-be RST, or a routing black hole. The signature is the lonely, repeating SYN. The timeout duration is a tell — a 30-or-so-second hang that ends in failure is the classic "DROP, not REJECT."

**SYN then RST.** The SYN leaves and an RST comes straight back, sub-millisecond. The host is up and reachable; nothing is listening on that port, so the kernel REJECTs with a reset. This is `connection refused`, and it is *good news* relative to the timeout case, because it means routing and the host are fine — you have a port or a process problem, not a network one. Section 2's worked example.

**ClientHello, no ServerHello.** TCP connected fine (you saw the handshake), the TLS ClientHello went out, and the server never replied with a ServerHello — or replied with a TLS alert and an RST. This is a TLS negotiation failure: a cert the client will not trust, an SNI the server does not recognize, or a protocol-version mismatch (client offers only TLS 1.3, server speaks only up to 1.2, or vice-versa with a hardened server rejecting old clients). Section 8 reads this in detail.

**Many retransmits / dup-ACKs.** The connection works but the same TCP segments are sent two, three, five times, and you see duplicate ACKs as the receiver keeps asking for the segment it is missing. The link is lossy. Throughput collapses and latency spikes because every loss costs a retransmission timeout. This is section 9, and it is the true cause behind a great many "the service is slow/flaky" reports that get misattributed to the application.

**TCP Zero Window.** The connection is established and data was flowing, then one side advertises `win=0`, meaning "my receive buffer is full, stop sending." The sender freezes until the receiver drains its buffer and re-advertises a non-zero window. This is not a network fault at all — it is the *receiving application* too slow to read what is arriving (a stalled consumer, a blocked event loop, a thread pool starved). The wire shows a network-shaped symptom (data stops flowing) whose cause is an application-shaped problem (the receiver is overwhelmed). Catching this saves you from chasing a phantom network issue when the real bug is a slow reader.

That last point generalizes: the wire shows you the *symptom's shape*, and several of these shapes have causes that live above or below the network. Zero-window is a slow app. Stale-DNS SYNs are a caching bug. RST-floods can be your own service closing sockets abruptly. The capture is not "proof it's the network" — it is proof of *what happened*, which often clears the network entirely and hands you a much more specific bug in your own stack.

## 8. TLS handshakes: reading cert, SNI, and version failures on the wire

HTTPS adds a TLS handshake on top of the TCP handshake, and TLS failures are some of the most confusingly-reported bugs in the business, because the application-level error is usually a vague "handshake failed" or "SSL error" or "certificate verify failed" with no indication of *which* part broke. The wire fixes that, because the early part of a TLS handshake is in plaintext — the ClientHello and ServerHello are not yet encrypted, so a capture shows you exactly what was offered, what was chosen, and where it stopped.

The handshake, in order: the client sends a **ClientHello** announcing its supported TLS versions, its cipher suites, and — critically — the **SNI** (Server Name Indication), the hostname it is trying to reach, in plaintext, so a server hosting many sites knows which certificate to present. The server replies with a **ServerHello** picking a version and cipher, followed by its **Certificate** chain. Key exchange establishes a shared secret, both sides send **Finished**, and from then on everything is encrypted. The figure below lays out the sequence, with the failure mode marked: a stall right after the ClientHello, with no ServerHello, is a server-side rejection.

![A left-to-right timeline of a TLS handshake showing ClientHello, ServerHello, the certificate chain, key exchange, and the encrypted Finished, with a final danger marker noting that a stall after the ClientHello with no ServerHello indicates a server-side failure](/imgs/blogs/its-the-network-packet-and-protocol-tracing-5.png)

You can read the plaintext part directly with `tshark`:

```bash
# Pull the SNI the client requested and the TLS version it offered:
tshark -r tls.pcap -Y 'tls.handshake.type == 1' \
  -T fields -e tls.handshake.extensions_server_name \
  -e tls.handshake.version

# See the server's chosen cipher and the cert it presented:
tshark -r tls.pcap -Y 'tls.handshake.type == 2' \
  -T fields -e tls.handshake.ciphersuite
tshark -r tls.pcap -Y 'tls.handshake.type == 11' -V   # the Certificate message
```

`tls.handshake.type == 1` is the ClientHello, `== 2` is the ServerHello, `== 11` is the Certificate. If you see type 1 but never type 2, the server rejected the client before answering — wrong SNI (the server had no cert for the name the client asked for), a version it would not negotiate, or no cipher in common. If you see type 1 and type 2 and then an `Alert` and an RST, the *client* rejected the *server's* certificate — usually an untrusted CA, an expired cert, or a hostname-mismatch where the cert's name does not cover the SNI.

For interactive TLS debugging, two tools beat reading raw packets. `openssl s_client` does the handshake for you and prints every detail of the cert chain and negotiation:

```bash
# Connect, send the right SNI, and dump the cert chain and negotiated version:
openssl s_client -connect api.example.com:443 -servername api.example.com 2>/dev/null \
  | openssl x509 -noout -subject -issuer -dates

# Just see what the server negotiates and whether the chain verifies:
openssl s_client -connect api.example.com:443 -servername api.example.com
```

The `-servername` flag sets the SNI — and forgetting it is itself a common bug, because a server that hosts many names will hand back its *default* cert (often the wrong one) when no SNI is sent, and then "the cert is wrong!" turns out to mean "you didn't tell the server which site you wanted." The output shows the subject, issuer, and the `notBefore`/`notAfter` dates, so an expired cert is immediately obvious, and the `Verify return code` at the bottom tells you whether the chain validated and why not.

`curl -v` is the other everyday tool, because it narrates the whole connection — DNS, TCP, TLS, and HTTP — in one go:

```bash
curl -v https://api.example.com/health
# Watch the lines: '* Trying 203.0.113.10:443...' (the IP it chose — check DNS!),
# '* SSL connection using TLSv1.3 / ...' (the version that won),
# '* Server certificate:' (subject, issuer, expiry, SAN),
# '> GET /health' and '< HTTP/2 200' (the actual request and response).

# For even more, the full handshake and headers:
curl --trace-ascii trace.txt https://api.example.com/health
```

`curl -v` is often the fastest first move for any HTTPS problem because it tells you, in one screen, the IP it resolved to (catching stale DNS), the TLS version and cipher it negotiated, the server's cert details (catching expiry and name mismatches), and the exact request and response headers (catching a proxy that mangled something). When `curl -v` succeeds but your application fails, the difference is usually in DNS (the app resolved to a different IP), in trust (the app uses a different CA bundle), or in SNI (the app's client library is not sending the name). Each of those is a specific, checkable hypothesis, and you got them all from one verbose flag.

## 9. Retransmissions, dup-ACKs, and the slow link that looks like a slow service

Now the bug that hides best behind "the service is just slow sometimes," because it produces intermittent latency with no errors in any application log: a lossy network link. When a TCP segment is lost in transit, the receiver never ACKs it, the sender's retransmission timer (RTO) eventually fires, and the sender resends. That retransmission costs you the full RTO — often a couple hundred milliseconds even on a fast network, because the RTO has a floor — and during that time the connection stalls. A connection that loses even one percent of its packets can see latency balloon, because every loss is a multi-hundred-millisecond penalty, and a request that needs many round trips compounds them.

The mechanism is worth understanding because it explains why a tiny loss rate causes a large latency spike. TCP guarantees in-order, reliable delivery, which means a lost segment *blocks everything behind it* — the receiver cannot deliver later segments to the application until the gap is filled, even if those later segments already arrived. This is head-of-line blocking. So one lost packet does not just cost its own retransmission; it stalls the entire stream until the resend arrives. The sender often learns about the loss faster than the RTO through **duplicate ACKs**: the receiver keeps ACKing the last in-order byte it got, and three dup-ACKs trigger a fast retransmit. So in a capture, the signature of a lossy link is a cluster of duplicate ACKs followed by a retransmission of the missing segment, repeated.

You find this with Wireshark's derived filters — you do not have to spot it by eye:

```bash
# Count retransmissions and dup-ACKs in a capture:
tshark -r slow.pcap -q -z io,stat,0,'COUNT(tcp.analysis.retransmission)tcp.analysis.retransmission'
tshark -r slow.pcap -Y 'tcp.analysis.retransmission' -T fields -e frame.time_relative -e tcp.seq

# Or just list every flagged anomaly with its type:
tshark -r slow.pcap -Y 'tcp.analysis.flags' \
  -T fields -e frame.time_relative -e tcp.analysis.retransmission -e tcp.analysis.duplicate_ack
```

If the retransmission count is non-trivial — say, dozens in a capture of a few hundred packets — you have a lossy link, and the slowness is the network, not your code. The next question is *where* the loss is, which you answer by capturing at multiple points (your host, the far host, a hop in between if you can) and seeing where the retransmissions appear versus where the original was clean. Loss that shows up only past a certain hop localizes the bad link.

#### Worked example: intermittent 504s and a 200ms RTO

An API gateway started returning intermittent `504 Gateway Timeout` errors — maybe one request in fifty, no pattern anyone could find. The upstream service's own logs were clean; it processed every request it *received* well within the timeout. The gateway team blamed the upstream, the upstream team blamed the gateway, and the metrics on both sides looked healthy because the failures were rare enough to vanish into the averages. A capture on the gateway, filtered to one of the slow connections, showed the truth:

```bash
sudo tcpdump -i any -w gw.pcap host 10.8.2.15 and port 9000
# let it run until a few 504s happen, then:
tshark -r gw.pcap -Y 'tcp.analysis.retransmission' \
  -T fields -e frame.time_relative -e ip.src -e tcp.seq
```

The output showed the same sequence numbers being retransmitted, with the retransmissions landing almost exactly 200 milliseconds after the originals — the RTO floor. On the slow requests, two or three retransmissions stacked up, adding 400 to 600 milliseconds, and a few of those pushed a request past the gateway's timeout, producing the 504. The upstream was innocent: it answered fast *when it got the request*, but the request (or its response) was being lost on a flaky link between the gateway and the upstream, and TCP's retransmission was eating the latency budget. The fix was at the infrastructure layer — one of the gateway nodes had a failing NIC dropping packets, and draining it took the retransmission count to zero. The measured result was clean: across 2,000 requests after the fix, the capture showed zero retransmissions to that upstream, p99 latency dropped from about 1,200 ms to about 80 ms, and the 504 rate went to zero. The number that mattered was the 200 ms — the RTO floor stamped on the wire told us this was retransmission latency and nothing else.

![A two-column before-and-after figure showing a flaky link where a lost segment triggers a 200ms retransmission timeout and the gateway returns a 504, contrasted with a fixed path that has zero retransmits over two thousand requests and p99 latency dropping from 1200ms to 80ms](/imgs/blogs/its-the-network-packet-and-protocol-tracing-8.png)

## 10. Socket state with ss and netstat: SYN-SENT, CLOSE_WAIT, and exhaustion

Before you even reach for a capture, the kernel already knows a great deal about every connection your process holds, and `ss` (the modern replacement for `netstat`) prints it. Every TCP connection is in a state — `SYN-SENT`, `ESTABLISHED`, `CLOSE_WAIT`, `TIME_WAIT`, and the rest — and a *pileup* in any one state is a specific diagnosis. You do not need a packet to see that ten thousand sockets are stuck in `CLOSE_WAIT`; `ss` will tell you in one line, and that pileup names a bug in your own code.

```bash
# Summary of socket counts by state (fast triage):
ss -s

# All TCP sockets with their states, no name resolution, with process info:
ss -tanp

# Just the ones stuck dialing (peer never answered):
ss -tan state syn-sent

# Count CLOSE-WAIT sockets (a leak signature):
ss -tan state close-wait | wc -l

# What's piling up in TIME-WAIT (usually benign, but can exhaust ports):
ss -tan state time-wait | wc -l
```

The figure below maps the four states you will care about most. Each is a different story, and the count in each cell is the diagnosis.

![A two-by-two grid of TCP socket states showing SYN-SENT meaning the peer is silent, ESTABLISHED meaning data is flowing, CLOSE-WAIT meaning the application forgot to close, and TIME-WAIT meaning a normal sixty-second wait](/imgs/blogs/its-the-network-packet-and-protocol-tracing-7.png)

**SYN-SENT** means the connection sent a SYN and is waiting for the SYN-ACK that never came. A pile of `SYN-SENT` sockets means your process is dialing peers that are not answering — the firewall-drop or wrong-port case from section 7, now visible without a capture. **ESTABLISHED** is healthy; data can flow. **CLOSE_WAIT** is the dangerous one and the most common real bug `ss` catches: it means the *remote* side closed the connection (sent a FIN), your kernel acknowledged it, and now the connection is waiting for *your application* to call `close()` — but your application never does. Each leaked `CLOSE_WAIT` holds a file descriptor. Enough of them and you hit the process's FD limit, after which every new connection and every file open fails, and the service falls over with `too many open files`. A climbing `CLOSE_WAIT` count is a near-certain "your code has a socket leak" — somewhere a connection is being abandoned without being closed, often in an error path that forgot its cleanup.

**TIME_WAIT** is the one people panic about unnecessarily. After a connection closes cleanly, the side that closed first holds the socket in `TIME_WAIT` for a couple of minutes (the famous 2×MSL) to make sure stray packets from the old connection do not get mistaken for a new one on the same port pair. This is *correct* behavior, not a leak. It only becomes a problem under very high connection churn — thousands of short-lived connections per second from one source — where you can exhaust the ephemeral port range and fail to open new outbound connections. The fix there is usually connection reuse (keep-alive) rather than fighting `TIME_WAIT`, which ties directly into the next section.

#### Worked example: the file-descriptor leak that took six hours to crash

A web service crashed roughly every six hours with `accept: too many open files`, then a supervisor restarted it and the clock reset. The crash was reproducible only by waiting, which made it miserable to debug — you could not iterate. But `ss` made the slow-motion failure visible in real time. A one-line watch showed the count climbing relentlessly:

```bash
watch -n 10 'ss -tan state close-wait | wc -l'
# 412 ... 530 ... 661 ... climbing ~120 every ten seconds toward the FD limit.
```

`CLOSE_WAIT` was the entire story. The remote clients were closing their connections (sending FINs), the kernel was acknowledging, and the application was never calling `close()` on its end — so each finished request leaked one FD. `ss -tanp` (with `-p` for process info) confirmed they all belonged to the service, and pointing the FD count at time-since-restart matched the six-hour crash cadence exactly: FD limit divided by the leak rate equaled the time-to-crash. The bug was an HTTP handler that returned early on a validation error without closing the upstream connection it had opened — an error path that skipped its cleanup. The fix was a `defer resp.Body.Close()` (Go) in the right place; after it, the `CLOSE_WAIT` count sat flat at single digits for days. The measured proof was the flat line: the count that had climbed by 120 every ten seconds now did not climb at all, and the six-hourly crash stopped. No packet capture needed — `ss` is a leak detector if you know which state to count.

## 11. Keep-alive vs connection-per-request, and the proxy that rewrote a header

Two more bug families round out the practical catalog, both of which a capture makes obvious and neither of which is visible from one side's logs.

**Keep-alive vs connection-per-request.** HTTP/1.1 connections are reusable: by default the connection stays open after a response so the next request can skip the TCP and TLS handshake, which is a large saving (a TLS handshake is one or two extra round trips). A misconfigured client or server that closes the connection after every request — sending `Connection: close`, or just FIN-ing — pays the full handshake cost on every single call, which under load looks like high latency and a flood of `TIME_WAIT` sockets from all the churn. In a capture this is unmistakable: a healthy keep-alive connection shows one handshake followed by many request/response pairs on the same stream; a connection-per-request pattern shows handshake, one request, FIN, handshake, one request, FIN, over and over. If you follow a few streams and each carries exactly one request before tearing down, you have found a keep-alive misconfiguration, and the fix (enable keep-alive, raise the idle timeout, increase the per-connection request limit) is usually a config line that cuts latency and socket churn dramatically. This connects to how load balancers manage connections; the deep treatment of L4 versus L7 balancing is in [load balancing from L4 to L7](/blog/software-development/system-design/load-balancing-from-l4-to-l7), where connection reuse and pooling at the balancer is a first-class concern.

**The proxy that rewrote or stripped a header.** When a request passes through a proxy, an API gateway, a service mesh sidecar, or a load balancer, that hop can modify it — add, drop, or rewrite headers — and a surprising number of "the request worked from my laptop but fails through the gateway" bugs are exactly this. The proxy strips an `Authorization` header it does not expect, or rewrites `Host`, or drops a custom header your service requires, or downgrades HTTP/2 to HTTP/1.1 and loses trailers. The way to catch it is the **capture-on-both-sides-and-diff** discipline: capture the request *as it enters* the proxy and *as it leaves*, and compare. What went in is not what came out, and the diff is the bug.

```bash
# Capture on the client-facing side of the proxy (what arrives):
sudo tcpdump -i eth0 -w ingress.pcap host CLIENT_IP and port 443
# Capture on the upstream side (what the proxy forwards):
sudo tcpdump -i eth1 -w egress.pcap host UPSTREAM_IP and port 8080
# Then follow each HTTP stream and diff the request headers:
tshark -r ingress.pcap -Y http.request -T fields -e http.request.line
tshark -r egress.pcap  -Y http.request -T fields -e http.request.line
```

For HTTPS, the request bytes are encrypted on the wire, so a plain `tcpdump` shows you the handshake but not the headers. To inspect the application layer of an HTTPS request, you need to decrypt it, and the clean way to do that in a debugging context is `mitmproxy` — a proxy you put in the path that presents *its own* certificate (which you trust on the client) so it can decrypt, show, and re-encrypt the traffic. It gives you a full readable view of every HTTP request and response, headers and bodies, flowing through it:

```bash
# Run mitmproxy and point your client at it (HTTPS_PROXY=http://localhost:8080),
# trusting mitmproxy's CA so it can decrypt:
mitmproxy --mode regular --listen-port 8080
# Or the scriptable, headless version for capturing to a file:
mitmdump -w flows.mitm
```

`mitmproxy` is the right tool when the bug is at the HTTP layer of an *encrypted* connection — a header being mangled, a body being truncated, a status code you do not expect — and you need to read the plaintext request and response. It is, by design, a man-in-the-middle, so you only ever point it at traffic you own and control; trusting its CA on a client you do not own is a security hole, which is exactly why it works as a debugging tool and exactly why you remove that trust when you are done. For raw socket-level visibility you stay with `tcpdump`; for application-layer visibility into HTTPS you reach for `mitmproxy`. They answer different questions and you keep both within reach.

## 12. HTTP/2, gRPC framing, and capturing inside a container or pod

Two modern wrinkles change how captures read, and you will hit both.

**HTTP/2 and gRPC multiplex many streams over one connection.** Where HTTP/1.1 used one request per connection (or serialized requests on a keep-alive connection), HTTP/2 interleaves many concurrent requests as *frames* on a single TCP connection, each request belonging to a numbered stream. This is great for performance and slightly harder to read in a capture, because what looks like one TCP connection actually carries dozens of logical requests woven together. gRPC runs on HTTP/2, so the same applies. Wireshark has an HTTP/2 dissector that reassembles the frames per stream, so you can still follow a single logical request — filter on `http2.streamid == N` — but you must reassemble at the HTTP/2 layer, not the TCP layer, because TCP-stream-following gives you the whole multiplexed bundle. The relevant failure mode here is that HTTP/2 has its own flow control *per stream* on top of TCP's, so a single slow stream can stall behind the connection's flow-control window, and you diagnose that by reading the `WINDOW_UPDATE` frames. For most debugging, though, the move is the same as ever: capture, find the connection, and let Wireshark's dissector split it back into the individual requests so you can read the one that failed.

**Capturing inside a container or pod is its own skill**, because the process you want to observe is in a different network namespace than your shell. A bare `tcpdump` on the host captures the host's interfaces, which may not show the pod's traffic the way the pod sees it (especially with overlay networking, NAT, and per-pod virtual interfaces). The clean trick is to run the capture *inside the target's network namespace* with `nsenter`:

```bash
# Find the container's PID, then enter just its network namespace and capture:
PID=$(docker inspect -f '{{.State.Pid}}' my-container)
sudo nsenter -t $PID -n tcpdump -i eth0 -w pod.pcap port 443

# In Kubernetes, the ephemeral-container approach attaches a debug sidecar
# that shares the target pod's network namespace:
kubectl debug -it my-pod --image=nicolaka/netshoot --target=my-app -- \
  tcpdump -i eth0 -w /tmp/pod.pcap port 443
```

`nsenter -t $PID -n` runs the following command in the network namespace of that PID, so `tcpdump` sees exactly the interfaces and traffic the containerized process sees — the pod's own `eth0`, with the pod's own IP, before any host-level NAT rewrites it. The Kubernetes `kubectl debug` ephemeral-container pattern does the same thing in a cluster: it injects a debugging container (the `netshoot` image bundles `tcpdump`, `tshark`, `dig`, `curl`, `ss`, and `mitmproxy`) that *shares the target pod's network namespace*, so your capture sees the application's traffic without you having to bake debugging tools into the production image. This matters enormously in practice because the discipline of "capture on the right interface" becomes "capture in the right namespace" in a containerized world, and capturing on the host when you needed to capture in the pod is the modern version of capturing on `eth0` when the traffic went out `eth1`. The traffic you want is real; you just have to be standing in the right place on the network to see it. For the broader picture of debugging across many services and namespaces, the system-design treatment in [debugging production at scale](/blog/software-development/system-design/debugging-production-at-scale) and [observability by design](/blog/software-development/system-design/observability-metrics-logs-traces-by-design) covers the tracing and correlation-ID layer that complements packet capture — capture tells you what crossed one wire; distributed tracing tells you the path across all of them.

## 13. The decision tree: from "no response" to the exact layer

Let us assemble everything into the routine you actually run when a call misbehaves, because the value of all these tools is realized only when you apply them in the right order. The figure below is the decision tree; the discipline is to ask one binary question at each fork and let the answer cut the search space in half — the same bisection that runs through the whole series, now applied to the network stack. If you want the general theory of this move, [binary-searching your bug with bisection](/blog/software-development/debugging/binary-search-your-bug-with-bisection) is its home; here we apply it to packets.

![A decision tree starting from no response, branching on whether a SYN left the machine, then on whether a SYN-ACK returned, then on whether TLS finished, with each leaf naming a specific layer such as DNS, firewall, or certificate failure](/imgs/blogs/its-the-network-packet-and-protocol-tracing-6.png)

Start at the top: **did a SYN leave the machine?** Capture `tcp[tcpflags] & tcp-syn != 0 and host TARGET` and reproduce. If no SYN appears, the problem is local — DNS, a local or egress firewall, a NetworkPolicy, an exhausted pool — and you go check `dig` and `iptables -L OUTPUT -v` and `ss -s`. You have eliminated the entire far side without touching it.

If a SYN did leave, ask: **did a SYN-ACK come back?** If the SYN repeats and nothing returns, a firewall is silently dropping it or you are dialing a wrong/blocked port — confirm by checking the route and the destination's listener. If an RST came back instantly, the port is closed (connection refused), and you go check what is actually listening on the target. If a SYN-ACK *did* come back and the connection established, you have proven the network path and the listener are fine, and the bug is higher up.

If the connection established and it is HTTPS, ask: **did TLS finish?** Filter `tls.handshake.type == 1` and `== 2`. ClientHello but no ServerHello means a server-side rejection — cert, SNI, or version — and you go run `openssl s_client -servername`. ClientHello, ServerHello, then an alert and RST means the client rejected the server's cert — check the CA bundle and the cert's expiry and SAN. If TLS finished and the request still failed, you are now genuinely at the application layer — a 5xx the server really generated, a header a proxy mangled (capture both sides and diff), or a slow response (read timeout) — and you reach for `mitmproxy` or the server's own logs, because you have *proven* the network delivered the bytes and the fault is in what the application did with them.

That is the whole method. Each fork is one cheap experiment that eliminates half the possibilities, and within four or five questions you have walked from "no response, no idea" to a named layer and a specific fix. The reason it works is the reason bisection always works: you are not guessing which of a dozen causes it is, you are *splitting* the dozen in half with each binary fact the wire hands you. The wire is what makes the questions binary — without a capture, "did a SYN leave?" is a guess; with one, it is a fact.

## 14. War story: how famous network bugs were actually caught

It helps to see how this kind of evidence cracked real, hard problems, because the discipline scales from a one-service config bug up to internet-wide incidents, and the move is always the same: capture the truth, compare it to the assumption, find the gap.

**Heartbleed (2014)** was a read-overflow in OpenSSL's implementation of the TLS heartbeat extension. A client could send a heartbeat request claiming a payload length far larger than the data it actually sent, and the vulnerable server would dutifully reply with that many bytes — reading past the end of the real payload into adjacent memory, leaking private keys, session cookies, and whatever else happened to be there. What makes it relevant here is that the *signature was on the wire*: a heartbeat request with a length field wildly larger than the actual payload, and a response far bigger than any legitimate heartbeat. Once the vulnerability was known, detection was literally a packet filter — you wrote a rule matching heartbeat responses whose size exceeded the request, and your capture infrastructure flagged every exploitation attempt. The bug lived in the protocol's framing, and the protocol's framing is exactly what a capture exposes. It is the cleanest example of "the bytes on the wire told the whole story" — the length field that lied was right there in plaintext for anyone reading the heartbeat extension.

**The TCP retransmission storms** that have taken down large services share a shape worth knowing because it is a trap you can build yourself. A small amount of packet loss on a congested link triggers retransmissions; the retransmissions add traffic to the already-congested link; the added traffic causes more loss; more loss causes more retransmissions. Without careful backoff this is a positive feedback loop — a congestion collapse — and it presents as a network that gets *worse* under load in a way that does not recover until you shed traffic. In a capture it is unmistakable: a wall of retransmissions and duplicate ACKs climbing as the link saturates. The fix in the protocol was TCP's congestion control (slow start, backoff), and the fix in *your* systems is the same idea one layer up — exponential backoff and jitter on retries, so a transient failure does not become a retry storm. A thundering herd of clients all retrying in lockstep after a blip will recreate congestion collapse at the application layer, and you will see it in the capture as a synchronized burst of SYNs. The capture is how you tell "the network is congested" (retransmissions everywhere) from "my clients are stampeding" (synchronized SYN bursts) — two very different fixes for symptoms that both feel like "the network melted."

**The split-horizon DNS outage** is a pattern many engineers have lived through even if it never made the news. A service works for months, then a DNS change — a migration, a failover, a CDN cutover — leaves some resolvers handing out the new IP and others (and long-running processes) still serving the old one. Half your fleet talks to the new endpoint, half talks to a ghost, and the symptom is maddeningly intermittent because it depends on *which resolver and which cached answer* each request happened to hit. The only thing that cuts through it is capturing on an affected host and seeing the SYNs go to the *old* IP while `dig` from a fresh resolver returns the *new* one — the exact capture-versus-`dig` mismatch from section 6, now at fleet scale. The lesson that engineers carry out of these is that DNS changes are deploys, with all the same caching and rollout hazards, and that "it's resolving fine for me" from your laptop proves nothing about what a pinned process on host 47 is actually dialing.

The thread through all three is that the bug was *legible on the wire* — a lying length field, a wall of retransmissions, a SYN to a stale IP — and invisible from any single component's logs. That is the standing argument for knowing how to capture and read packets: when the components disagree, the wire is the impartial witness.

## 15. How to reach for packet capture (and when not to)

Packet capture is powerful and it is not free, so be deliberate about when it earns its keep, because reaching for `tcpdump` when a one-line `curl -v` would have answered the question is its own kind of inefficiency.

**Reach for it when the components disagree.** The moment two teams are blaming each other and each is looking only at their own logs, a capture is the fastest way to end the argument with a fact neither side can wave away. This is its highest-value use: it is the impartial third party.

**Reach for it when the symptom is "no response" or "timeout" with no error to read.** When the application's error is a vague timeout and nothing logged a cause, the capture is often the *only* instrument that can tell you whether a packet even left, which is the question that splits the whole search in half.

**Reach for it for intermittent, infrastructure-shaped latency** — the flaky 504s, the occasional slow request with no application-side explanation — because retransmissions and zero-windows are invisible above the transport layer and a capture is how you see them.

**Do not reach for it when a cheaper tool answers the question.** If `curl -v` reproduces the failure, it tells you the IP, the TLS version, the cert, and the headers in one screen — start there, and only escalate to a capture if `curl` succeeds while your app fails (which localizes the difference) or if the failure is below the HTTP layer. If `ss` shows ten thousand `CLOSE_WAIT` sockets, you have your bug without capturing a single packet. If `dig` returns the wrong IP, you are done. The capture is the high-resolution instrument; do not bring it out for a question a quick command already settles.

**Do not capture blindly on a busy production host.** A `tcpdump` with no filter on a high-traffic interface can fill a disk in minutes and add measurable CPU load, and on a truly saturated box the capture itself can drop packets and lie to you. Always filter tightly (`host X and port Y`), always bound the capture (`-c` or `-G`/`-W` rotation), and prefer to reproduce on a single connection you control rather than firehosing prod.

**Mind the privacy and security of what you capture.** A packet capture of unencrypted traffic contains everything — credentials, tokens, personal data, request bodies. Treat `.pcap` files as sensitive: do not leave them on shared hosts, do not paste them into tickets without scrubbing, and never run `mitmproxy` against traffic you do not own, because decrypting someone else's TLS is exactly the attack the whole system is designed to prevent. The same power that makes capture a great debugger makes it a great wiretap; keep it pointed only at your own traffic, and delete the captures when you are done.

**Capture in the right place.** Half of all "the capture showed nothing" frustration is capturing on the wrong interface or in the wrong network namespace. On a container, `nsenter` into the pod's namespace; in Kubernetes, use an ephemeral debug container that shares the pod's network. If you suspect a specific hop (a proxy, a load balancer), capture on *both* sides of it and diff. The wire only helps if you are standing where the bytes actually flow.

| Tool | What it answers | Reach for it when | Cost / caution |
| --- | --- | --- | --- |
| `curl -v` / `--trace` | DNS, TLS, headers, status in one shot | first move for any HTTP(S) failure | none; start here |
| `dig` | what a name resolves to, and TTL | "is DNS handing back the right IP?" | none; check the app's resolver |
| `ss` / `netstat` | socket states and counts | leak (CLOSE_WAIT), stuck dial (SYN-SENT), exhaustion | none; instant triage |
| `tcpdump` | exactly what crossed the wire | components disagree; timeout with no cause | disk/CPU on busy hosts; filter tightly |
| Wireshark / `tshark` | follow a stream, derived anomalies | read a conversation, count retransmits | needs the `.pcap`; learn display filters |
| `openssl s_client` | the TLS handshake and cert chain | "is it the cert, SNI, or version?" | none; remember `-servername` |
| `mitmproxy` | decrypted HTTP layer of HTTPS | a header/body bug inside encrypted traffic | only on traffic you own; it's a MITM |

## 16. Key takeaways

- **Stop asking "is it the network?" and start asking "which layer, and what's the signature?"** A request crosses DNS, TCP, TLS, and HTTP, and a failure at any low layer masquerades as a high-layer error. The capture tells you which step actually happened.
- **The most powerful capture result is the absence of a packet.** No SYN on the wire means the problem is local — DNS, firewall, pool exhaustion — and you have eliminated the entire far side in one observation.
- **RST versus FIN is a one-bit diagnosis with huge leverage.** An instant RST after a SYN is a closed port (host is up, nothing listening); an RST mid-connection is something aborting; a FIN is a graceful close that may be sooner than you wanted.
- **`ss` is a leak detector.** A climbing `CLOSE_WAIT` count is your code forgetting to `close()`; a pile of `SYN-SENT` is a peer that never answers; `TIME_WAIT` is usually fine. You catch FD exhaustion before it crashes you, no capture required.
- **Retransmissions and a 200ms RTO on the wire are the real cause behind a great many "the service is slow" reports.** A lossy link is invisible above the transport layer; `tcp.analysis.retransmission` makes it count itself.
- **Read the plaintext part of the TLS handshake.** ClientHello-with-no-ServerHello is a server-side cert/SNI/version rejection; an alert after the ServerHello is the client rejecting the cert. `openssl s_client -servername` and `curl -v` answer it in one command.
- **Capture in the right place.** The right interface, the right network namespace (`nsenter`, `kubectl debug`), and both sides of any proxy you suspect. A capture on the wrong interface shows nothing and teaches you nothing.
- **Diff what went into a proxy against what came out.** A stripped or rewritten header is invisible from either side alone and obvious from a both-sides capture; for HTTPS app-layer inspection, decrypt with `mitmproxy` on traffic you own.
- **Use the cheapest instrument that answers the question.** `curl -v`, `dig`, and `ss` settle most cases without a single captured packet; bring out `tcpdump` when the components disagree or the symptom is a silent timeout.
- **Treat captures as sensitive and turn every observation into a falsifiable hypothesis.** The wire is the impartial witness in the observe → reproduce → hypothesize → bisect → fix → prevent loop; it converts "the network is acting up" into "here is the SYN, here is the RST, the port is closed."

## 17. Further reading

- **The intro to this series — [Stop guessing: the scientific method of debugging](/blog/software-development/debugging/stop-guessing-the-scientific-method-of-debugging)** — the observe → reproduce → hypothesize → bisect → fix → prevent loop that packet capture serves as its highest-resolution `observe` step.
- **[Binary-searching your bug with bisection](/blog/software-development/debugging/binary-search-your-bug-with-bisection)** — the general theory behind the "did a SYN leave?" decision tree; each binary question on the wire halves the search.
- **A planned sibling on syscall tracing** (`strace`/`ltrace`/`bpftrace`) — what a process *really* does at the system-call boundary, the layer just below the socket, complementing the wire-level view here; and a planned sibling on debugging across service boundaries, which extends the both-sides-of-a-proxy discipline to a whole request path.
- **[Load balancing from L4 to L7](/blog/software-development/system-design/load-balancing-from-l4-to-l7)** — how balancers manage connections, keep-alive, and header rewriting, the source of many of the proxy and connection-per-request bugs in section 11.
- **[Observability: metrics, logs, traces by design](/blog/software-development/system-design/observability-metrics-logs-traces-by-design)** and **[debugging production at scale](/blog/software-development/system-design/debugging-production-at-scale)** — distributed tracing tells you the path across all the wires; packet capture tells you the truth about one. Use them together.
- **The `tcpdump` man page and the Wireshark User's Guide** — the canonical references for BPF capture filters and display filters respectively; the filter languages are small and repay an hour of study.
- **Wireshark's TCP Analysis documentation** — how `tcp.analysis.retransmission`, `duplicate_ack`, and `zero_window` are *derived* from the stream, so you trust what the dissector flags.
- **TCP/IP Illustrated, Volume 1 by W. Richard Stevens** — the deep reference for the handshake, retransmission, congestion control, and every flag and state you will read off the wire.
- **The OpenSSL `s_client` documentation and Brendan Gregg's networking-tracing material (eBPF/bpftrace)** — for TLS handshake debugging and for the kernel-side tracing that sits just below packet capture.
