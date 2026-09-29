---
title: "BGP: How the Internet Decides Where Your Packet Goes"
date: "2026-09-29"
publishDate: "2026-09-29"
description: "Follow a route from an AS announcement through local policy to a forwarding decision, then diagnose withdrawals and anycast failures."
tags:
  [
    "networking",
    "distributed-systems",
    "bgp",
    "internet-routing",
    "autonomous-systems",
    "anycast",
    "route-selection",
    "route-withdrawal",
    "network-diagnostics",
  ]
category: "software-development"
subcategory: "Networking"
author: "Hiep Tran"
featured: true
readTime: 41
image: "/imgs/blogs/bgp-how-the-internet-decides-where-your-packet-goes-1.webp"
---

Your service has healthy pods in two regions. DNS returns the expected address. A client on one ISP reaches the edge in its own city; another client reaches a distant edge and times out. The application deployment did not change. The load balancer's pool did not change. The route to the address changed somewhere between those clients and your network.

![The client obtains an edge address from DNS, while BGP determines which network can deliver packets to that edge](/imgs/blogs/bgp-how-the-internet-decides-where-your-packet-goes-1.webp)

The diagram above is the mental model. DNS supplies an address, but application packets do not travel through the resolver. Routers forward those packets toward the address. At each network boundary, a locally chosen route determines the next network. The edge and its anycast announcement are the focus of this post. The later L4 balancer, L7 proxy, sidecar, application, and backend still matter, but they cannot repair packets that never reach the edge.

In [the series introduction](/blog/software-development/networking/what-actually-happens-when-you-curl-a-url), we follow a request through every layer. Here we isolate one decision: how independent networks learn that an IP prefix exists and choose a path toward it. The answer is Border Gateway Protocol (BGP), plus the commercial and operational policies wrapped around it. BGP is a control-plane protocol, a way to exchange reachability. It does not carry your HTTP request, measure your request latency, or continuously optimize the route for your user.

The practical promise is narrower and more useful than a routing certification syllabus. By the end, you should be able to distinguish a bad DNS answer from a missing route, explain why the shortest AS path may lose, recognize what an anycast withdrawal actually changes, and collect evidence from more than one vantage point before blaming your service.

> A destination address is a request to the network. It is not a promise about the path that will carry the packet.

## 1. One prefix, many independent decisions

An autonomous system (AS) is a network under one routing policy, identified on the public Internet by an AS number. An AS can be an access ISP, cloud provider, content network, enterprise, or transit network. This is a policy boundary, not necessarily one building or one router. A large AS can span continents; a small AS can have a single external connection. [RFC 4271, published January 2006](https://www.rfc-editor.org/rfc/rfc4271), defines BGP as an inter-AS reachability protocol. Its central object is a route to a destination prefix, accompanied by path attributes. A prefix is an address block such as the documentation range `203.0.113.0/24`.

An announcement says, in effect, "I can reach this prefix through the path described by these attributes." A withdrawal says, "Do not keep using the route I previously announced to you." Neither statement says that the application behind the prefix is healthy. Neither statement proves that the route will be accepted. The neighbor can reject the announcement through an import filter, prefer a different accepted route, or fail to resolve the announced next hop. The neighbor can also decline to export its chosen route onward.

That distinction matters when an incident dashboard says "BGP up." It may mean a TCP session between two routers is established, while the desired prefix is absent, filtered, not selected, or not installed for forwarding. Conversely, an application can be unhealthy while the prefix remains globally reachable. The control plane and the service plane are different observations.

### What BGP is not measuring

BGP carries reachability claims and policy attributes. It does not send a probe for every prefix, compute the user's current round-trip time, or ask whether an HTTP handler returned a good response. Its best-path decision uses the information and policy available to a particular router. That is why a route can be stable while performance deteriorates on a congested link, and why a route can change even when application servers remain healthy. Monitoring needs both route state and request outcomes.

This also changes how we interpret "fastest route." A path with fewer ASes is not necessarily faster. The count says little about fiber length inside each AS, queueing at a handoff, or where TLS and application processing happen. If a customer reports a slow request, inspect the latency components and compare source networks. A BGP change is a plausible cause when the source-specific path changed at the same time; it is not proof until the address, prefix, and vantage points line up. The [latency budget post](/blog/software-development/networking/the-latency-budget-speed-of-light-serialization-and-queueing) provides the method for separating propagation, serialization, queueing, and server time.

Nor is BGP the route lookup executed for every packet. Routers program a forwarding table from selected routes and other local sources. Packet forwarding then uses that table at line rate. The distinction explains why a control-plane view and a packet symptom can briefly disagree during a change. It also tells us which question a tool can answer: a BGP neighbor view shows learned or advertised reachability, while a forwarding-table query shows the route currently used by one device for one destination. Keep both snapshots when investigating a transient failure.

### Prefixes are the destination keys

The destination key is a prefix, not a hostname. DNS may map `api.example.test` to `203.0.113.42`, but BGP never compares the hostname. Once the client has an address, the local router performs a longest-prefix match against its forwarding table. A `/25` covering that address is more specific than a `/24`, so it wins at that forwarding lookup even if the `/24` has an attractive BGP path. Only after the prefix is fixed does the route-selection question among candidates for that prefix make sense. Confusing these two levels produces bad incident explanations.

Consider `203.0.113.42`. The first 25 bits of its address match `203.0.113.0/25`, and the first 24 bits also match `203.0.113.0/24`. The `/25` covers the lower half of the `/24`, so a router with both entries uses `/25` for this destination. `203.0.113.200` does not match that lower `/25`, so it continues to use `/24`. These are documentation addresses, and this is a derived address-membership example, not an observation of a public route. The [address and subnet post](/blog/software-development/networking/addresses-subnets-nat-and-the-packets-two-identities) works through the bit arithmetic.

### The AS path is not the packet's street map

`AS_PATH` records AS-level path information and helps prevent loops. It is useful evidence of which ASes a route announcement traversed. It is not a packet-level traceroute. Within one AS, internal routing can use several routers, links, tunnels, or points of presence. Traffic engineering can choose a different exit for a given ingress. A traceroute may omit hops or produce misleading answers when routers do not answer probes, load balancing changes probe paths, or return traffic takes another way. Treat the AS path and the forwarding path as related maps at different scales.

The data plane forwards packets using its installed forwarding information base (FIB). BGP maintains and selects control-plane routes; an implementation then installs eligible routes into the routing table and FIB, subject to next-hop resolution and interactions with other routing sources. The operator sees these as separate questions: Did I learn the prefix? Did I accept it? Did I choose it? Did it become a usable forwarding entry? Did packets actually travel through the expected exit?

## 2. Transit, peering, and why a route might never reach you

![Customer, provider, and peer relationships constrain which routes an AS exports](/imgs/blogs/bgp-how-the-internet-decides-where-your-packet-goes-2.webp)

The graph is deliberately an economic map. A provider sells transit: it agrees to carry the customer's traffic toward other destinations and to make the customer's routes reachable elsewhere. Peering networks exchange traffic under an agreement that usually concerns their own networks and customers. The exact contract varies. BGP itself does not encode the price or force a peering policy. Import and export configuration implements the agreement.

Suppose network A buys transit from P and peers with B. If A learns a route from its customer, A can often export that route to its provider and peer: they can reach A's customer through A. If A learns a route from P, advertising it to B would offer B free passage through A to P's other customers. Most networks do not intend that. A common policy therefore exports customer-learned routes broadly while exporting peer- or provider-learned routes primarily to customers. This is the familiar "valley-free" intuition, a policy convention rather than a universal law of BGP.

This gives us a better mental model than "every router knows every route." Each AS sees only what its neighbors chose to export, after its own filters. The globally visible graph is assembled from local bilateral exchanges. An apparently excellent physical fiber path may never be a candidate at all because one AS will not export the route across that commercial boundary. Conversely, a configuration error can export routes across a boundary where they should have stopped. [RFC 9234, published May 2022](https://www.rfc-editor.org/rfc/rfc9234), specifies BGP roles and the Only-to-Customer (OTC) attribute to help detect and prevent some route leaks. It is a control, not a claim that all interdomain policy is enforced by the protocol.

### Route leak versus hijack versus ordinary policy

These words describe different failures. A route leak commonly means an AS propagates a route beyond the intended scope of its relationship. An origin hijack means an AS originates a prefix it should not announce. A more-specific hijack can direct traffic for a subset of a larger legitimate block toward an unintended network. An AS can also make a perfectly legitimate policy choice that happens to produce a path an application operator dislikes. These can look similar from a browser: timeout, slow response, or certificate failure. The route origin, AS path, prefix length, and time sequence separate them.

Do not call every odd traceroute a leak. First ask whether the affected address belongs to a prefix with a changed origin or a newly visible more-specific. Then compare route collectors from several networks. A single local traceroute cannot establish that the rest of the Internet made the same choice. If the route remains stable while only one ISP changes behavior, the next question is that ISP's internal exit policy or its upstream's export choice.

### A provider is not a global route planner

It is tempting to draw one map and ask for the globally shortest path. No AS has a complete, authoritative, current view of every private link, contract, capacity constraint, maintenance window, and service health signal. Even with that view, networks have different objectives. A transit customer may value cost, an access ISP may value direct interconnection, and a content network may value staying on its own backbone. BGP allows each network to apply local policy to the routes it receives. The resulting path is a composition of decisions made by different owners, often at different times.

This is the first reason your packets can take a route nobody chose deliberately end to end. Each participant chose its own next step. The assembled journey emerged from those local choices. The same pattern appears in [service discovery and load balancing](/blog/software-development/microservices/service-discovery-and-load-balancing), but there the control plane belongs to one organization. On the public Internet, the control plane crosses organizations that cannot impose a shared objective.

## 3. From UPDATE message to forwarding entry

A BGP session runs between neighbors and exchanges reachability with UPDATE messages. [RFC 4271](https://www.rfc-editor.org/rfc/rfc4271) describes routes, path attributes, the decision process, and withdrawal in those messages. The important operational pipeline is: receive an update, apply import policy, test whether the next hop can be resolved, select an eligible route for the prefix, export according to policy, and install the selected forwarding result. The exact implementation details and tie breakers vary by router and configuration.

A route is not automatically usable just because it appears in a BGP table. The NEXT_HOP attribute must lead to an actual reachable next hop in the local routing system. Import policy might reject a private or unexpected prefix. A router might keep several candidate paths but advertise only its selected one to a neighbor under ordinary BGP operation. Internal BGP distribution, route reflectors, multipath features, and add-path extensions change which routes particular routers can see. When debugging, record which router and which table a command showed.

Here is a synthetic example for one prefix. These numbers are intentionally illustrative policy values, not a benchmark or a prescribed configuration. The values only demonstrate the ordering of two comparisons in a common implementation: a higher local preference can override a shorter AS path. Cisco's [best-path documentation, checked September 2026](https://www.cisco.com/c/en/us/support/docs/ip/border-gateway-protocol-bgp/13753-25.html) describes that vendor's ordering. [RFC 4271](https://www.rfc-editor.org/rfc/rfc4271) leaves room for local policy, so do not treat a vendor's full tie-break list as one universal BGP algorithm.

| Candidate to `203.0.113.0/24` | Local preference | AS path length | Eligible next hop | Result | Source |
| --- | ---: | ---: | --- | --- | --- |
| Via paid transit | 200 | 4 ASes | Yes | Selected in this illustrative policy | Synthetic example; Cisco best-path order linked above |
| Via peer | 100 | 2 ASes | Yes | Available but not selected | Synthetic example; Cisco best-path order linked above |

The transit route wins here because the illustrative local preference is higher. The path-length comparison never gets to overrule that earlier policy choice. Reverse the local preferences and the peer route wins, provided both remain eligible. This does not say transit is generally preferred over peering. Many operators deliberately prefer customer routes, then peer routes, then transit routes, because of economics and traffic engineering. The particular values and ordering are policy. The lesson is that path length is not a latency measurement and need not be the first deciding attribute.

![An imported route with preferred local policy wins over a shorter AS path in this illustrative selection](/imgs/blogs/bgp-how-the-internet-decides-where-your-packet-goes-3.webp)

### The path attributes worth reading first

`AS_PATH` lists AS-level traversal and supports loop detection. A shorter path may be preferred after earlier policy checks, but fewer AS numbers do not guarantee fewer router hops, less propagation delay, or lower congestion. An AS could span a continent while several ASes interconnect in one building. An operator can prepend its own AS number to make one advertised path appear longer to some neighbors. That is a traffic-engineering hint, not a command the remote network must obey.

`LOCAL_PREF` expresses preference inside an AS. It is commonly used to choose an outbound exit and is not advertised to external BGP peers as a transitive command. It lets a network say, "For this destination, prefer this class of neighbor or this exit." If two paths differ in local preference, staring at AS path length alone is the wrong diagnosis. Ask which import policy set the value and on which routers.

`NEXT_HOP` names the address to reach for the route. If that next hop cannot be resolved, an attractive route is unusable. An internal underlay failure can therefore remove a BGP path from service without any change in the external destination's announcement. This is the bridge from control-plane claims to physical forwarding. In an incident, a BGP table can still show a route while the FIB or recursive next-hop lookup exposes the actual break.

`MULTI_EXIT_DISC`, often shortened to MED, can signal a preferred ingress among links between neighboring ASes. It is typically compared under specific conditions and local policy can ignore or reinterpret it. An application owner should not assume a low MED on its route will override another AS's commercial preference. `ORIGIN` and implementation-specific tie breakers may matter later. Communities can signal policy requests, but a community has effect only when the receiving network has agreed to act on it. Reading attributes in context beats memorizing a universal ordered list.

### Two selection layers, two different questions

![A more-specific prefix changes the forwarding match even while the less-specific route still exists](/imgs/blogs/bgp-how-the-internet-decides-where-your-packet-goes-4.webp)

The figure separates the two decisions that are often collapsed into one phrase, "BGP chose the route." First, the FIB matches the destination to the most specific installed prefix. Second, for a given prefix, control-plane policy determined which candidate route was installed. A more-specific prefix can therefore divert traffic for part of a block without defeating the existing `/24` in the same-prefix best-path contest. The `/24` continues to handle addresses outside the new `/25`.

Take the derived documentation-address example again. Before a `/25` appears, both `203.0.113.42` and `203.0.113.200` match `203.0.113.0/24`. After `203.0.113.0/25` becomes an installed route, only `.42` moves to the `/25`. This is arithmetic on prefix membership; it is not evidence that a public operator accepted a particular route. The distinction was central to Cloudflare's account of the [June 24, 2019 route leak](https://blog.cloudflare.com/how-verizon-and-a-bgp-optimizer-knocked-large-parts-of-the-internet-offline-today/): their reported `104.20.0.0/20` was split into `104.20.0.0/21` and `104.20.8.0/21`, and those more-specific routes were propagated when they should not have been. Those exact public prefixes and the date come from Cloudflare's incident report.

A route must be accepted and installed before longest-prefix match can act on it. Prefix filtering, route-origin validation, maximum-prefix limits, and export policy can stop a bad route earlier. Route-origin validation checks whether an origin AS is authorized for a prefix, including allowed maximum length where a Route Origin Authorization (ROA) exists. It does not prove that every intermediate AS relationship is legitimate or that the application is healthy. See [RFC 6811, published January 2013](https://www.rfc-editor.org/rfc/rfc6811) for the origin-validation procedure. Think of origin validation and relationship-based leak prevention as complementary checks on different claims.

## 4. Why the path can be surprising even when every network behaves correctly

You cannot infer the global route from your own edge router's preference. Your network controls its own outbound selection. Remote access networks control their own exits. Transit providers control what they import and export. The destination operator can influence inbound routing by changing advertisements, prefix lengths within filtering norms, locations, communities honored by providers, and AS-path presentation. Influence is not ownership of the remote decision.

One common surprise is hot-potato routing. An AS may hand traffic to another network at a nearby exit to minimize distance on its own backbone. The neighbor may then carry the packet a long distance. Another AS might keep traffic on its backbone and hand it off closer to the destination. Both can be reasonable local policies. Neither optimizes the client's end-to-end latency by itself. A change to an internal metric can move the chosen egress even while external BGP announcements remain stable. Conversely, a changed external announcement can move traffic while internal metrics are unchanged.

A second surprise is asymmetry. The forward route from client to server and the return route from server to client are chosen by different networks for different destination prefixes. A traceroute from your server is not a reverse traceroute of the client's packets. This matters when a firewall expects symmetric paths, when a capture sees requests but not responses, and when an RTT increase is blamed on the visible forward path. Use measurements from both ends or independent vantage points when possible.

A third surprise is aggregation. A network may announce a covering prefix while a more-specific route is present only inside its own domain. That keeps the global route table smaller, but it hides internal placement from external observers. It also means a site failure inside the aggregate can become a black hole if the covering route remains advertised and the network has no internal path to the service. Announcing every internal change globally would create its own churn and scale problem. The safe design is to make the relationship between external reachability and internal delivery explicit.

A fourth surprise is time. Each AS learns, filters, selects, and advertises changes independently. There is no instantaneous global commit. Two collectors can briefly show different views without either being wrong about its own vantage point. A route that disappeared at your border may still be present in a neighbor's FIB until its update is processed; an alternate route may already be selected elsewhere. Avoid promising an exact Internet-wide reconvergence time from one lab or one dashboard. The interval depends on topology, policy, timers, implementation, and the availability of an alternate path.

### A diagnostic hierarchy

Start with the user's resolved IP. Confirm the answer, resolver, and time of lookup. Then identify the covering prefix and origin seen from several BGP vantage points at the relevant time. Ask whether a more-specific prefix appeared or a known prefix disappeared. Next compare AS paths and next-hop behavior by source network. Only then use traceroute and application metrics to narrow the physical segment and service impact. That sequence avoids interpreting a slow TLS handshake as proof of BGP change or a BGP change as proof of application failure.

The commands are intentionally read-only on a production host:

```bash
# Confirm the address and resolver response seen by this host.
dig +noall +answer example.com A

# See this host's first-hop route to a specific observed address.
ip route get 203.0.113.42

# Scope a path probe to the destination; interpret missing hops carefully.
traceroute -n 203.0.113.42

# Check whether connection timing changed independently of DNS timing.
curl --silent --show-error --output /dev/null \
  --write-out 'remote=%{remote_ip} dns=%{time_namelookup} connect=%{time_connect} first=%{time_starttransfer}\n' \
  https://example.com/
```

`ip route get` answers only the host's local forwarding choice. It does not reveal upstream BGP policy. `traceroute` samples one direction with probe-specific behavior and incomplete responses. `dig` says what address was returned, not whether that address is globally reachable. `curl` shows user-visible timing but cannot alone assign the delay to a particular AS. Keep those scopes separate in your incident notes. For the packet-level local routing mechanics, see [ARP, switching, routing, and ECMP](/blog/software-development/networking/how-a-packet-gets-there-arp-switching-routing-and-ecmp).

## 5. A withdrawal is a distributed state change

<figure class="blog-anim">
<svg viewBox="0 0 900 260" role="img" aria-label="A prefix withdrawal reaches neighboring autonomous systems in sequence; each removes its previously learned route" style="width:100%;height:auto;max-width:900px">
<style>
.bgp17-card{fill:var(--surface,#f3f4f6);stroke:var(--border,#d1d5db);stroke-width:2}
.bgp17-label{font:600 18px ui-sans-serif,system-ui;fill:var(--text-primary,#1f2937);text-anchor:middle}
.bgp17-small{font:14px ui-sans-serif,system-ui;fill:var(--text-secondary,#6b7280);text-anchor:middle}
.bgp17-link{stroke:var(--border,#d1d5db);stroke-width:4}
.bgp17-route{fill:#a5d8ff;stroke:var(--text-primary,#1f2937);stroke-width:1}
.bgp17-withdraw{fill:#ffc9c9;stroke:var(--text-primary,#1f2937);stroke-width:1}
@keyframes bgp17-hop{0%,10%{transform:translateX(0);opacity:0}15%{opacity:1}35%{transform:translateX(220px)}60%{transform:translateX(440px)}85%,95%{transform:translateX(660px);opacity:1}100%{transform:translateX(660px);opacity:0}}
@keyframes bgp17-fade1{0%,32%{opacity:1}40%,100%{opacity:.15}}
@keyframes bgp17-fade2{0%,57%{opacity:1}65%,100%{opacity:.15}}
@keyframes bgp17-fade3{0%,82%{opacity:1}90%,100%{opacity:.15}}
.bgp17-moving{animation:bgp17-hop 10s ease-in-out infinite}
.bgp17-r1{animation:bgp17-fade1 10s steps(1,end) infinite}
.bgp17-r2{animation:bgp17-fade2 10s steps(1,end) infinite}
.bgp17-r3{animation:bgp17-fade3 10s steps(1,end) infinite}
@media (prefers-reduced-motion:reduce){.bgp17-moving,.bgp17-r1,.bgp17-r2,.bgp17-r3{animation:none}.bgp17-moving{opacity:0}}
</style>
<line class="bgp17-link" x1="150" y1="110" x2="250" y2="110"/>
<line class="bgp17-link" x1="370" y1="110" x2="470" y2="110"/>
<line class="bgp17-link" x1="590" y1="110" x2="690" y2="110"/>
<rect class="bgp17-card" x="30" y="60" width="120" height="100" rx="12"/>
<rect class="bgp17-card" x="250" y="60" width="120" height="100" rx="12"/>
<rect class="bgp17-card" x="470" y="60" width="120" height="100" rx="12"/>
<rect class="bgp17-card" x="690" y="60" width="120" height="100" rx="12"/>
<text class="bgp17-label" x="90" y="105">Origin AS</text>
<text class="bgp17-label" x="310" y="105">Transit AS</text>
<text class="bgp17-label" x="530" y="105">Access AS</text>
<text class="bgp17-label" x="750" y="105">Client AS</text>
<text class="bgp17-small" x="90" y="135">withdraws</text>
<text class="bgp17-small" x="310" y="135">reselects</text>
<text class="bgp17-small" x="530" y="135">reselects</text>
<text class="bgp17-small" x="750" y="135">reselects</text>
<rect class="bgp17-route bgp17-r1" x="264" y="184" width="92" height="24" rx="5"/>
<rect class="bgp17-route bgp17-r2" x="484" y="184" width="92" height="24" rx="5"/>
<rect class="bgp17-route bgp17-r3" x="704" y="184" width="92" height="24" rx="5"/>
<text class="bgp17-small" x="310" y="229">learned route</text>
<text class="bgp17-small" x="530" y="229">learned route</text>
<text class="bgp17-small" x="750" y="229">learned route</text>
<circle class="bgp17-withdraw bgp17-moving" cx="90" cy="35" r="15"/>
<text class="bgp17-small" x="445" y="252">Control-plane update, not a packet path or a time scale</text>
</svg>
<figcaption>A withdrawal is processed neighbor by neighbor; each AS removes or replaces its learned route according to local policy.</figcaption>
</figure>

A route withdrawal begins at one speaker, not at every router simultaneously. The speaker sends an UPDATE withdrawing reachability it had previously advertised to a neighbor. The neighbor removes that candidate route, runs its own decision process, and may advertise a new selected route or a withdrawal to its neighbors. Each receiving AS applies its own import, selection, and export logic. That is the mechanism the animated figure shows. The moving signal means a control-plane update is being processed in sequence. It does not mean packets physically follow the animation's red line or that every AS takes an equal amount of time.

If an alternate route exists and is eligible, a network can select it. If not, the prefix may disappear from that network's forwarding view. In either case, packets already in flight can be dropped, reordered, or delivered to a different edge during the transition. BGP convergence is not an atomic switch. In a service incident, a successful probe from one geography does not prove that another geography has converged, and a failed probe does not prove that the prefix was withdrawn everywhere.

It is useful to write down four distinct states for the same prefix at one vantage point:

1. **Announced by origin.** The originating AS has decided to advertise it.
2. **Visible to a collector.** At least one observer has received an announcement from its peers.
3. **Selected by a network.** A specific AS has accepted and preferred one candidate route.
4. **Forwarded successfully.** Packets reach the destination and get useful service.

Those states can diverge. An origin can announce a route that a transit provider filters. A collector can see a route that your ISP does not select. A route can be selected and still point toward a dead application. A healthy application can become unreachable because its route was withdrawn. An incident timeline should name which state each observation supports.

A withdrawal can be intentional. An operator may stop advertising a site's anycast prefix when the site's upstream connectivity or service dependencies are unhealthy. That can move new traffic toward other sites, if alternate announcements remain reachable and each remote network selects one. The same mechanism can amplify a fault. If a health check depends on a shared backbone, then a backbone failure can cause many otherwise running edge locations to withdraw simultaneously. That is exactly why the Meta incident in the next section belongs in a BGP explanation, even though the initiating fault was a backbone maintenance command.

One should not equate withdrawal with immediate failover. Remote networks need to receive and process the change. They may retain an alternate path, receive one later, or have none. An already established TCP connection can fail if the destination shifts to an edge without that connection's state. Application retries may mask a brief event or multiply load on the surviving site. The [timeouts and retries post](/blog/software-development/site-reliability-engineering/timeouts-retries-and-backoff-done-right) owns retry policy; here the key is that the network route and the connection state can change on different schedules.

### Oscillation and damping

A route that repeatedly appears and disappears creates churn. Each transition triggers work in neighbors and may change forwarding. Operators therefore care about the health signal that controls an announcement. A probe that is too sensitive can turn a small, local application blip into a wider route flap. A probe that is too insensitive can keep advertising a dead site and black-hole traffic. Hysteresis, multiple independent checks, and staged withdrawal reduce the chance that one transient observation changes global reachability. They introduce a recovery trade-off: requiring sustained health before re-advertisement can delay return to the preferred site.

During planned maintenance, a blunt session reset may produce an avoidable interruption while neighbors search for alternatives. [RFC 8326, published March 2018](https://www.rfc-editor.org/rfc/rfc8326) specifies a graceful BGP session shutdown procedure using a community and reduced local preference so traffic can move before the session disappears, where participating networks honor it. This is an example of policy coordinating a transition, not a promise of zero packet loss. An application team need not configure the BGP session, but it should ask whether edge maintenance drains connections and routes in a compatible order.

## 6. Anycast: one address, several possible edges

![Several sites announce the same anycast prefix while independent client networks choose their own route](/imgs/blogs/bgp-how-the-internet-decides-where-your-packet-goes-5.webp)

Anycast means multiple locations advertise reachability for the same IP prefix. A client sends to one destination address, and interdomain routing determines which advertising location receives the traffic from that client's network. It is natural to call this "nearest site," but physical distance is not the selection rule. Policy, available paths, interconnection, and internal egress choices determine the route. The nearest site by kilometers can lose to a farther site with a preferred relationship or a better handoff.

The useful application property is that the destination address can stay constant while traffic moves. A DNS answer can remain cached and valid. If one site withdraws the route, another site advertising the same prefix may become the selected path. That can make anycast a powerful front-door design for stateless services such as DNS and for edge services that terminate and distribute requests. It also creates sharp requirements for stateful traffic. If packets in an established TCP flow move to an edge with no corresponding socket state, the flow can reset or time out. A new connection can succeed while the old one fails. A service dashboard that counts only successful new connections may hide that user-visible disruption.

Health checks for anycast must ask the right question. "Can this edge answer a local health request?" is weaker than "Can this edge deliver the user's request through all required dependencies?" Yet a check tied too tightly to a shared backend can withdraw every edge at once when the backend fails. Sometimes that is correct because serving an error at an advertised site is worse than steering to healthy sites. Sometimes it is catastrophic because there are no healthy alternatives. The operator needs a dependency map and an explicit behavior for total dependency failure, not a single Boolean named `healthy`.

A local edge withdrawal also changes traffic elsewhere. Surviving sites receive additional connections, cache misses, and backend calls. If those sites have little spare capacity, failover can turn a regional impairment into a global overload. The safe answer is not merely "announce from two sites." It is to test site loss, define capacity headroom, and verify connection and retry behavior. The [L4 to L7 load-balancing article](/blog/software-development/system-design/load-balancing-from-l4-to-l7) handles how an edge divides traffic after packets arrive. BGP anycast determines which edge receives them in the first place.

### Anycast and DNS are separate control planes

DNS can return the same anycast address to everyone, yet different source networks can reach different sites. DNS can also return different addresses by geography while each returned address is itself anycast. These mechanisms compose, but their failure modes differ. A stale DNS answer affects which destination address the client sends to. A BGP withdrawal affects whether a route to an address exists and which site receives packets for it. During an incident, record both the DNS answer and the route evidence rather than using "DNS issue" as a catch-all.

The [DNS production guide](/blog/software-development/networking/dns-in-production-ttls-caching-layers-and-stale-answers) explains cache layers and TTL limits. A DNS TTL does not bound BGP convergence. A BGP update does not flush the client's resolver cache. When designing failover, draw the two timelines separately, then draw where they meet at the connection attempt. That is especially important when a service uses DNS to select a regional VIP and anycast to select the edge location within that VIP.

## 7. Two public cases that expose the mechanism

### Meta, October 4, 2021: reachable servers with withdrawn DNS routes

Meta's [incident-owner write-up, published October 5, 2021](https://engineering.fb.com/2021/10/05/networking-traffic/outage-details/), states that during routine maintenance a command intended to assess global backbone capacity unintentionally disconnected the backbone. An audit system was supposed to stop such a command, but a bug prevented that. That was the trigger. The backbone disconnection cut connectivity among data centers and between server infrastructure and the Internet.

The BGP part was a second mechanism. Meta's smaller facilities answered authoritative DNS queries. Their health logic withdrew the BGP advertisements for those DNS-server addresses when the facilities could not reach the data centers. With the backbone down, the facilities declared themselves unhealthy and withdrew those routes. According to Meta, the DNS servers were still operational, yet became unreachable from the Internet. The user-visible symptom was that Meta services could not be found or reached normally. Treating this as a simple "DNS server crash" misses the route withdrawal that made a running DNS server unreachable.

The blast radius came from coupling an edge route's health to a shared backbone state. The recovery blockers were separate: Meta reported that normal data-center access and out-of-band access were unavailable, and internal tools depended on DNS. Engineers had to go onsite. This is why a routing control plane deserves an independent management path and why emergency access should not depend entirely on the service whose route may disappear. The transferable diagnostic test is to compare authoritative service health with external reachability of its announced prefix. If the process is running but remote vantage points have no route, restarting the process is not the first repair.

The published report does not provide a universal BGP convergence duration for this event, so none is asserted here. Nor does it say BGP was the original trigger. The maintenance command and audit failure initiated the outage; withdrawal of DNS advertisements multiplied the reachability failure. That causal separation matters when reviewing controls. A stronger command audit addresses the trigger. Independent access and careful health-to-announcement coupling address recovery and blast radius.

### Cloudflare, June 24, 2019: a more-specific route escaped its boundary

Cloudflare's [incident-owner report, published June 24, 2019](https://blog.cloudflare.com/how-verizon-and-a-bgp-optimizer-knocked-large-parts-of-the-internet-offline-today/), describes a route leak involving an optimizer that divided received prefixes into more-specific routes. The report uses Cloudflare's own `104.20.0.0/20` becoming `104.20.0.0/21` and `104.20.8.0/21` as an example. According to Cloudflare, those more-specifics traveled from DQE Communications through Allegheny Technologies to Verizon, which propagated them more broadly without the filtering that should have stopped them. Traffic for affected prefixes was drawn toward networks not prepared to carry it.

This is a different failure from Meta's withdrawal. Here an unwanted announcement became visible, and prefix specificity made it attractive at forwarding time. It was not a demonstration that the leaked route had lower latency, nor that every network's same-prefix BGP preference chose it. The cause lay in export and import controls, plus the fact that an installed more-specific route overrides a covering prefix for matching addresses. Cloudflare reported loss of about 15 percent of its global traffic at the worst point of the event. That percentage is Cloudflare's incident observation, not a general benchmark for route leaks.

The transfer lesson is to test route policy at both boundaries. The exporting network should not propagate routes outside its authority or intended relationship. The receiving provider should apply prefix and relationship filters and monitor unexpected more-specifics. From an application team's vantage point, the quick test is whether the affected source networks see a new route to a more-specific prefix that healthy source networks do not. The next post, [BGP incidents, hijacks, and leaks](/blog/software-development/networking/bgp-incidents-hijacks-and-the-day-facebook-withdrew-itself), examines these incident classes and routing security controls in depth.

## 8. Work two routes all the way through

A useful routing explanation should survive a concrete destination and a stated policy. The next examples are synthetic. Their prefixes are reserved for documentation, their policy scores are illustrative, and no latency or availability result is attributed to a real network. They let us keep three questions separate: which prefix matches the address, which candidate route is selected for that prefix, and whether the resulting next hop can deliver a packet.

### Example A: the shorter AS path loses

Suppose AS 65010 learns two routes to `203.0.113.0/24`. One comes from a peer and advertises AS path `65020 65030`, a length of two AS numbers. The other comes from a transit provider and advertises `65040 65050 65060 65030`, a length of four. The AS numbers here are from a private-use range for illustration. If AS 65010 assigns the transit route local preference 200 and the peer route 100, and both next hops are reachable, a common policy ordering selects transit before comparing AS path length. This is exactly the sort of result the selection matrix illustrated earlier.

What could make that sensible? Perhaps the peer link has a contractual scope or capacity concern, while the transit provider carries a guaranteed service class for this prefix. Perhaps the local preference was set by a broad policy that now deserves review. We cannot infer which from the path attributes alone. The correct diagnostic question is which import rule assigned local preference and whether that rule was intended for this route. The route's selected status tells us what the router did, not why the business rule was written.

Now change only local preference: assign both routes 100. In a common implementation, with earlier comparisons tied and no exceptional policy, the shorter peer AS path can win. That says nothing definite about user RTT. One AS in the two-hop path could contain a long internal route; four adjacent ASes could interconnect over a short physical distance. To decide whether the selected path is slower, measure from affected sources at the transport or application layer and compare relevant vantage points. BGP attributes are route-control evidence, not a stopwatch.

Finally, suppose the peer's NEXT_HOP becomes unreachable. Even a high local preference cannot make an unusable next hop forward packets. A network may fall back to the transit route, withdraw the prefix, or show a mismatch between the BGP table and the installed FIB while state settles, depending on its topology and implementation. The operational check is to inspect route validity and recursive next-hop resolution, then compare the installed forwarding entry. A route line copied from a BGP table is incomplete evidence without those fields.

### Example B: a more-specific route bypasses that contest

Keep AS 65010's selected `/24` transit route. Add a valid, installed route for `203.0.113.0/25` through a different neighbor. The address `203.0.113.42` falls in the lower half of the `/24`, so the `/25` now wins its FIB lookup. The address `203.0.113.200` falls in the upper half, so it still uses the `/24`. The local preference values assigned to candidates for the `/24` do not make the `/24` more specific. This is why a route leak that creates and propagates more-specific prefixes can change traffic while operators continue to see a perfectly good aggregate route.

The arithmetic is deterministic. A `/24` fixes the first 24 address bits and contains 256 IPv4 addresses. A `/25` fixes one more bit and contains 128. Those counts are derived from ${2}^{32-p}$ addresses for an IPv4 prefix length $p$: for $p=24$, ${2}^{8}=256$; for $p=25$, ${2}^{7}=128$. The `Run it yourself` lab uses the exact lower-half case and inspects the kernel's resulting lookup. It does not need a public BGP session to prove the FIB rule.

This is also why "our route is still visible" does not settle a reachability incident. Visibility of the aggregate is compatible with a damaging more-specific. A route collector query should look for *covering and covered* prefixes around the affected address, not only an exact string match on the aggregate you expected to announce. The origin AS and AS path of the more-specific then matter for deciding whether it was authorized, leaked, or part of a planned traffic-engineering change.

### Example C: an anycast site withdraws but one flow still fails

Suppose two sites advertise one prefix and a client network currently selects site A. Its router forwards a new TCP connection to A, where A creates connection state. A's health system then withdraws the route. After a path change, the same destination address may lead the client's next packet to site B. B can be healthy for new connections yet have no state for the established one. The old connection can reset or time out while a fresh connection succeeds. That pair of observations is not contradictory. It is a signature of routing and connection state moving on different schedules.

The alternative is that no surviving site advertises an eligible route from that client's network. Then even a new connection fails before it reaches an edge. Yet another possibility is that the old route remains selected for a while at that network, so neither packet has moved yet. The application team needs to distinguish these states with source-specific probes. Test a fresh connection and an existing connection separately during a planned withdrawal. Note the remote IP, source network, and connection result. If the old flow breaks but the new one succeeds, focus on state continuity and retry behavior. If both fail and the prefix disappears from relevant route views, focus on announcement and alternate reachability.

These examples show why application architecture and network policy must be reviewed together. [Service discovery](/blog/software-development/networking/service-discovery-from-dns-to-registries-to-xds) decides which service endpoint a process calls. DNS decides which address it learns. BGP decides whether and where the interdomain route to that address leads. An L4 or L7 balancer chooses among endpoints after the packet reaches its front door. Each layer can change independently. A successful change at one layer does not imply the others are ready.

## 9. A production diagnostic sequence that respects vantage point

When a user reports that one network cannot reach your service, resist the urge to change the app before identifying the first broken boundary. The following sequence turns a vague connectivity report into bounded evidence. It is written for an application or SRE operator who may not control BGP configuration but can collect useful observations for the network team.

**Freeze the destination.** Record the hostname, exact resolved address, address family, resolver, time, and affected source network. A browser may race IPv4 and IPv6 or reuse an existing connection. Testing a different address than the user used is not a reproduction. Run `dig` for the relevant record and use `curl --resolve` when you need to hold a hostname, TLS name, and address constant for a direct application test. Keep the command and output together in the incident log.

**Check local reachability without overclaiming.** `ip route get` shows the host's local route to the destination. It can catch an incorrect local policy route, VPN, or container namespace table. It does not reveal the ISP's chosen AS path. `ping` can show that ICMP replies arrive or fail, but packet filters may treat ICMP differently from TCP, and a reply does not prove the application is healthy. Use a TCP or HTTPS probe to the service port as well.

**Compare independent BGP observers.** A route collector or looking glass can show whether a prefix and origin were visible at a particular observer. Compare several observers near affected and unaffected networks. Note the collector's peer, observation time, full prefix, origin AS, and AS path. A route collector is not an oracle for every forwarding router. If the relevant peer did not export a route to the collector, absence in that collector alone is weak evidence. A confirmed more-specific with an unexpected origin or path is stronger evidence, especially when its start time matches the user's symptoms.

**Trace the packet path with caveats.** Run traceroute or MTR from both affected and healthy source networks when permission and reachability allow. Vary probes if necessary, because ICMP, UDP, and TCP probes may be treated differently. A star in the output can mean a router did not answer, not that it dropped transit traffic. The hop that first changes is a lead; it is not automatically the faulty owner. A route might change at an AS boundary while the first visible traceroute difference occurs inside an upstream network.

**Separate route, edge, and origin.** If a route is absent from multiple relevant vantage points, investigate announcement and filtering. If the route exists but TCP connect fails, investigate forwarding, ACLs, capacity, and edge listeners. If connect succeeds but first byte is slow, the request has progressed further than BGP route acquisition. It may still have taken a bad path, but server and proxy timing now matter too. The [senior engineer's network mental model](/blog/software-development/networking/the-senior-engineers-network-mental-model) brings these discriminating observations together across the series.

![A diagnostic tree separates missing route evidence from edge reachability and application response](/imgs/blogs/bgp-how-the-internet-decides-where-your-packet-goes-6.webp)

The tree's branches are questions, not automatic blame assignments. Route visibility varies by observer. TCP success does not certify the app. HTTP success from one source does not certify all sources. Each step rules out some explanations and suggests the next measurement. That is more useful during an outage than a single red or green "network" dashboard tile.

### What an application team can safely change

Application teams can make routing decisions easier to operate without directly editing router policy. Keep an inventory of service VIPs, announced prefixes, origin ASes, expected anycast sites, and the health dependency that controls each announcement. Test a site withdrawal during a controlled exercise and record what happens to new and established connections. Budget capacity for surviving sites. Ensure an independent way to reach edge and core management planes. Export per-site service health separately from route-announcement state. Give the network team a time series of affected addresses and source ASes rather than only URLs and screenshots.

Changes to BGP export policy, prefix length, ROAs, and upstream communities require coordination with the people who own the network and its contracts. A broad more-specific announcement can shift traffic much farther than a local load-balancer setting. A route withdrawal can remove the only path for a region. Those changes deserve staged rollout, independent external observation, and a rollback path that does not require DNS or the affected backbone to function. The operational principle is simple: make the control plane observable from outside the system it controls.

## 10. Change routing without turning a local repair into a global outage

A routing change has a larger audience than the configuration file where it starts. Before changing an announcement, write the intended effect in terms of prefix, origin, neighbors, and source networks. "Move traffic to the other site" is too vague. A useful statement is: "Stop advertising this exact prefix from site A to its upstreams, keep site B advertising it, and verify that probes from these independent access networks establish new connections to B." This makes the post-change checks falsifiable. It also reveals whether the plan depends on DNS, anycast, or both.

Validate the prefix set before the change. A typo in mask length can create a more-specific that wins more traffic than intended or an aggregate that covers addresses you do not serve. Validate origin authorization and maximum length against ROAs where used, and ask providers what prefix-length filters they apply. The route can be valid in your router and still disappear at a peer's import filter. For a new prefix, arrange external observation before the announcement so the team can see which vantage points accepted it. For a withdrawal, arrange an independent access path to the devices that must restore it.

Stage the change where possible. Start with one site or one neighbor, check the route view from several external networks, then broaden. That sequence is not always available during an emergency, but it should be the default for planned work. Watch both the control plane and the service plane: advertised prefix, selected origin and AS path, connect success, first-byte time, and load on surviving edges. If new connections succeed but established sessions fail, record that explicitly. A rollback trigger based only on fresh HTTP success can miss a real user problem.

The rollback plan should identify a known-good advertisement and the authority to restore it. It should not require logging in through the same DNS name or backbone that might disappear. Meta's October 2021 account is a concrete warning about coupled recovery paths, not an argument to keep advertising unhealthy services forever. The right behavior under total dependency failure depends on whether any alternative location can serve. Decide it before the incident, exercise it, and keep the control-plane decision observable from outside.

For a suspected leak, the safest immediate application action may be to preserve evidence and escalate to network operators and providers rather than changing service configuration. The path may be wrong before packets arrive at your edge. Moving pods or restarting load balancers cannot repair an upstream export-policy error. Conversely, if the prefix and route views remain stable while only an origin dependency fails, a route-policy change may widen the blast radius. The diagnostic sequence above exists to keep the repair at the layer that produced the symptom.

The compact review checklist is: exact prefix, origin, authorized maximum length, intended upstreams, intended export scope, alternate site capacity, established-flow behavior, external observers, and out-of-band rollback. Each item corresponds to a failure mode in the mechanisms we have traced. It is not a guarantee that every Internet path will obey your intent. It gives the team a way to notice quickly when local intent and remote selection diverge.

## Run it yourself

### Question

Can a newly installed more-specific route change a host's forwarding result for one address while the covering route remains installed for another? This tests the FIB consequence that makes a more-specific BGP announcement powerful. It does **not** emulate BGP sessions, remote export policy, or Internet convergence. Those are upstream control-plane conditions required before a route can appear in this FIB.

### Preconditions

Use the Linux `netlab` namespaces `c` and `s` from [the setup in post 1](/blog/software-development/networking/what-actually-happens-when-you-curl-a-url). This experiment needs `iproute2`, `sudo` privilege to change routes inside namespace `c`, and the canonical `c0` to `s0` link. It changes only two documentation-prefix routes inside `c`. Do not run the route mutations on a production interface or host table. On macOS, run the series lab inside the privileged Linux environment described in post 1. `198.51.100.0/24` and `203.0.113.0/24` are documentation ranges, not live destinations for this experiment.

Preflight checks the namespace, link, route, tool version, and absence of prior experiment state. The lab does not require a server process because it inspects forwarding decisions rather than delivering application bytes. Check the usual netlab server listener as a reminder of the stable topology, but do not start or modify it for this route test:

```bash
set -euo pipefail
ip -Version
sudo ip netns list
sudo ip -n c -br address show dev c0
sudo ip -n s -br address show dev s0
sudo ip -n c route get 10.77.0.2
sudo ip netns exec s ss -lnt '( sport = :8080 )' || true
sudo ip netns exec c tc -s qdisc show dev c0
for prefix in 203.0.113.0/24 203.0.113.0/25; do
  if sudo ip -n c route show "$prefix" | grep -q .; then
    echo "Stop: $prefix already exists in namespace c" >&2
    exit 1
  fi
done
```

Read: `c0` is `10.77.0.1/30`, `s0` is `10.77.0.2/30`, and `ip route get 10.77.0.2` selects `c0`. `ss` may show a listener on port `8080` if you left the introductory lab running; its presence is not required. `tc -s qdisc` is recorded to show this experiment does not change queueing. Expected: the two documentation routes are absent; otherwise the script stops rather than replacing existing namespace state.

### Baseline

Add one covering route in the isolated client namespace. The route's next hop is the existing server-side veth address. We are only asking the kernel which entry it would choose; `s` is not configured to forward these documentation addresses, so do not treat this as an end-to-end connectivity test.

```bash
set -euo pipefail
sudo ip -n c route add 203.0.113.0/24 via 10.77.0.2 dev c0
sudo ip -n c route show 203.0.113.0/24
sudo ip -n c route get 203.0.113.42
sudo ip -n c route get 203.0.113.200
```

Read: both `route get` results name `via 10.77.0.2 dev c0`. Expected: both addresses use that next hop because both lie in the `/24`, and no more-specific route exists. The numeric addresses here are documentation examples. The expectation is a deterministic prefix match inside this namespace, subject to the preflight confirming no competing route.

### Apply one change

Install a more-specific blackhole route for the lower half of the same block. A blackhole is intentionally used so the changed forwarding result is unmistakable. It stands in for a newly selected, bad more-specific FIB entry; it is not a model of BGP's full decision process.

```bash
set -euo pipefail
sudo ip -n c route add blackhole 203.0.113.0/25
sudo ip -n c route show 203.0.113.0/25
```

Read: `ip route show` names `blackhole 203.0.113.0/25`. Expected: the `/24` still exists. The only changed variable is the additional `/25` entry.

### Compare

```bash
set -euo pipefail
if sudo ip -n c route get 203.0.113.42; then
  echo 'Unexpected: lower-half address remained forwardable' >&2
  exit 1
else
  echo 'Expected: lower-half address hit the blackhole /25'
fi
sudo ip -n c route get 203.0.113.200
sudo ip -n c route show 203.0.113.0/24
sudo ip -n c route show 203.0.113.0/25
```

Read: the `.42` lookup fails with a blackhole or invalid-route diagnostic, while the `.200` lookup still names `via 10.77.0.2 dev c0`. Expected: this split is exact because `.42` is in `203.0.113.0/25` and `.200` is outside it but inside `203.0.113.0/24`. The diagnostic wording and exit status formatting can differ by `iproute2` version, so the script tests success versus failure instead of comparing literal English output. This is the same forwarding principle that made the 2019 more-specific leak consequential, but it is a controlled local demonstration, not a reproduction of that public incident.

### Reset

```bash
set -euo pipefail
sudo ip -n c route del blackhole 203.0.113.0/25
sudo ip -n c route del 203.0.113.0/24 via 10.77.0.2 dev c0
sudo ip -n c route show 203.0.113.0/24
sudo ip -n c route show 203.0.113.0/25
sudo ip -n c route get 10.77.0.2
```

Read: neither documentation route remains, while the canonical route to `10.77.0.2` still selects `c0`. The reset touches only the two routes this experiment added. For a production system, the safe read-only analogue is `ip route get <observed-destination>` on the relevant host or namespace, paired with BGP route evidence from the network team or a looking glass. A host lookup cannot by itself tell you why an upstream AS selected or advertised its route.

## What to keep in your incident notebook

A BGP explanation is strongest when it connects a control-plane observation to a user-visible path change without skipping layers. Record the source network, resolved destination address, exact prefix, route origin, observer and timestamp, any newly visible more-specific, and the first application operation that failed. Then state which claim each artifact supports. A BGP collector can establish what one collector learned. A local FIB query can establish one host's first forwarding decision. A TCP probe can establish whether a connection succeeded from one source. None is a substitute for the others.

When there is no route change, keep working down the path. An edge can be overloaded, an L4 balancer can reject new connections, an L7 proxy can route to an unhealthy upstream, and an application can hang on a backend. The first diagram leaves those components visible for a reason. When there *is* a route change, do not stop at the word BGP. Ask whether the new path was imported, selected, exported, installed, and actually used, and whose policy made each decision. That turns an opaque Internet problem into a sequence of questions with owners and measurements.

For further primary reading, start with [RFC 4271](https://www.rfc-editor.org/rfc/rfc4271) for the reachability and decision model, [RFC 9234](https://www.rfc-editor.org/rfc/rfc9234) for route-leak roles, and the dated [Meta](https://engineering.fb.com/2021/10/05/networking-traffic/outage-details/) and [Cloudflare](https://blog.cloudflare.com/how-verizon-and-a-bgp-optimizer-knocked-large-parts-of-the-internet-offline-today/) incident accounts for two different ways routing state can make an otherwise plausible service destination unreachable.
