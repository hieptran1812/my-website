---
title: "BGP incidents: Hijacks, leaks, and the day Facebook withdrew itself"
date: "2026-09-29"
publishDate: "2026-09-29"
description: "Learn to distinguish stolen routes, leaked routes, and self-withdrawal from the signals an application team can actually observe."
tags:
  [
    "networking",
    "distributed-systems",
    "bgp",
    "route-hijack",
    "route-leak",
    "rpki",
    "dns",
    "incident-response",
    "internet-routing",
  ]
category: "software-development"
subcategory: "Networking"
author: "Hiep Tran"
featured: true
readTime: 41
image: "/imgs/blogs/bgp-incidents-hijacks-leaks-and-the-day-facebook-withdrew-itself-1.webp"
---

Your origin is healthy. The load balancer reports green backends. The application logs show no incoming requests. Customers in one region say the site is gone, while a colleague in another region can still open it. This is the moment when an application engineer tends to ask whether DNS is broken. That is a useful question, but it is one layer too high if the resolver cannot reach an authoritative server because the route to that server has changed.

![The path from a client and resolver through interdomain routing to an application edge](/imgs/blogs/bgp-incidents-hijacks-leaks-and-the-day-facebook-withdrew-itself-1.webp)

The diagram above is the mental model: a request crosses networks you do not operate before it reaches your edge, and even the DNS lookup crosses that boundary. Border Gateway Protocol (BGP) is the control plane that tells autonomous systems (ASes), independently operated networks, how to reach IP prefixes. The application packet follows a forwarding decision made from those announcements. If the announcement is wrong, missing, or propagated beyond its intended scope, a healthy service can disappear, become slow, or lead a resolver to the wrong place.

This post is the incident companion to [how BGP chooses a path](/blog/software-development/networking/bgp-how-the-internet-decides-where-your-packet-goes). We will use five dated cases to answer a more operational question: which routing failure happened, what should we have measured, and which control could have reduced its blast radius? The cases are deliberately different. Pakistan Telecom's YouTube announcement and the Route 53 attack involved unauthorized routes. Meta withdrew its own DNS routes after a backbone failure. Rogers and KT show how an internal routing mistake can take down a provider without anyone stealing an Internet prefix. Calling all five a “BGP hijack” would conceal the mechanism.

For the whole request path, start with [what happens when you curl a URL](/blog/software-development/networking/what-actually-happens-when-you-curl-a-url). DNS resolution and service discovery get their own treatment in [DNS as a distributed database](/blog/software-development/networking/dns-the-distributed-database-your-request-starts-in) and [DNS caching in production](/blog/software-development/networking/dns-in-production-ttls-caching-layers-and-stale-answers). Here we stay at the route boundary: where a name server or application address becomes reachable, unreachable, or reachable through the wrong network.

## 1. Four failures that look like one outage

**The user sees a timeout; the routing system sees an announcement and a policy decision.** Separate those views before assigning blame. A timeout is an outcome, not a diagnosis. A TCP SYN timeout can follow packet loss in the access network, a firewall rule, an overloaded destination, or a bad route. A DNS timeout can follow a dead authoritative server, but it can also follow a healthy authoritative server whose address is no longer globally reachable.

![Four routing failure classes with their different triggers and packet outcomes](/imgs/blogs/bgp-incidents-hijacks-leaks-and-the-day-facebook-withdrew-itself-2.webp)

The figure separates four classes. A **prefix hijack** occurs when a network originates an IP prefix it is not authorized to originate. A more-specific prefix can win the destination lookup even when the legitimate origin still announces its aggregate. A **route leak** is an announcement carried beyond its intended policy scope. [RFC 7908, published in June 2016](https://www.rfc-editor.org/rfc/rfc7908.html), defines the latter by the policy violation, not by whether the resulting path is malicious. A **self-withdrawal** occurs when the legitimate origin retracts a route, often because a health mechanism says the destination cannot serve traffic. An **internal redistribution failure** occurs inside a provider when route information crosses an internal protocol boundary without the intended filter or containment.

Those categories can overlap in a real event. A route that should have remained local can leak and become a competing, unauthorized origin for a destination prefix. The operational question is which boundary failed first. Did someone originate a prefix they did not own? Did a peer export a legitimate route to a place its policy forbade? Did the rightful origin remove its own advertisement? Or did an internal router swallow more routing state than it could handle? The answer determines whether you page your transit provider, inspect your own health-driven announcements, or stop an internal change.

| Failure class | Control-plane change | Likely packet outcome | First useful evidence |
| --- | --- | --- | --- |
| Unauthorized origin | Another AS announces your prefix or a more-specific part | Packets follow the false origin in networks that accept it | Public route collector origin and prefix history |
| Route leak | An AS exports a learned route beyond intended peers | Detour, congestion, interception opportunity, or loss | New AS path and unexpected export relationship |
| Self-withdrawal | Your AS or provider retracts a legitimate route | Destination unreachable unless an alternative survives | Withdrawal event, origin health and announcement logs |
| Internal redistribution error | Internal protocol receives routes outside its designed scope | Router overload or wrong internal forwarding | Change diff, routing-process counters and core health |

The table is a diagnostic map, not a claim that every network will show the same symptoms. BGP is distributed and policy-driven. Two client networks may choose different paths to the same address. Some resolvers cache an answer and continue to work briefly. Others ask an authoritative server immediately. A partial incident is not evidence against a routing hypothesis.

### The route and the packet are different objects

A BGP announcement says, roughly, “this AS can reach this prefix through this AS path.” A router evaluates the route under local policy and installs a forwarding next hop. An IP packet contains a destination address. It does not carry the BGP AS path. When packets vanish, the application's logs on the destination can remain perfectly quiet because the packets never crossed the final border.

Suppose a legitimate network advertises an illustrative prefix `203.0.112.0/23`, and a different network announces `203.0.113.0/24`. The /24 covers one half of that /23 address block. For a destination inside the /24, longest-prefix matching chooses a matching /24 forwarding entry over a /23 entry, even if the /23 comes from the rightful origin. This is a derived address-range example using documentation space, not a measured event. Local BGP policy decides whether the /24 is accepted and which competing /24 is selected; longest-prefix match applies after routes become forwarding entries. “The shorter AS path wins” is therefore an unsafe shortcut. Prefix specificity and policy can dominate the symptom.

This distinction matters because a route collector observes announcements, while a probe observes actual packet delivery. A collector can show that a suspicious route was visible at its peers. It does not prove every router on Earth installed it, or that a particular customer packet traversed it. Conversely, a failing probe does not identify the faulty announcement. During response, keep the control-plane evidence and the data-plane evidence side by side.

### Withdrawal has a propagation time

A legitimate operator may withdraw a route because a site is unhealthy. Neighboring ASes then update their own views and may advertise the change onward. The following animation shows the causal order, not a fixed duration. The initial state is a route held by each network. The terminal state is no route along this particular chain; another path, if present, could still carry traffic.

<figure class="blog-anim">
<svg viewBox="0 0 860 240" role="img" aria-label="A BGP withdrawal propagates from the origin through transit networks; each network loses the route after receiving the update" style="width:100%;height:auto;max-width:860px">
<style>
.bgpi-box{fill:var(--surface,#f3f4f6);stroke:var(--border,#d1d5db);stroke-width:2}.bgpi-line{stroke:var(--border,#d1d5db);stroke-width:3}.bgpi-text{font:600 17px ui-sans-serif,system-ui;fill:var(--text-primary,#1f2937);text-anchor:middle}.bgpi-small{font:14px ui-sans-serif,system-ui;fill:var(--text-secondary,#6b7280);text-anchor:middle}.bgpi-x{font:700 24px ui-sans-serif,system-ui;fill:#dc2626;text-anchor:middle;opacity:0}@keyframes bgpi-one{0%,12%{opacity:0}18%,100%{opacity:1}}@keyframes bgpi-two{0%,35%{opacity:0}42%,100%{opacity:1}}@keyframes bgpi-three{0%,58%{opacity:0}65%,100%{opacity:1}}@keyframes bgpi-four{0%,81%{opacity:0}88%,100%{opacity:1}}.bgpi-a{animation:bgpi-one 12s ease-in-out infinite}.bgpi-b{animation:bgpi-two 12s ease-in-out infinite}.bgpi-c{animation:bgpi-three 12s ease-in-out infinite}.bgpi-d{animation:bgpi-four 12s ease-in-out infinite}@media (prefers-reduced-motion:reduce){.bgpi-a,.bgpi-b,.bgpi-c,.bgpi-d{animation:none;opacity:1}}
</style>
<line class="bgpi-line" x1="178" y1="110" x2="258" y2="110"/><line class="bgpi-line" x1="378" y1="110" x2="458" y2="110"/><line class="bgpi-line" x1="578" y1="110" x2="658" y2="110"/>
<rect class="bgpi-box" x="58" y="68" width="120" height="84" rx="12"/><rect class="bgpi-box" x="258" y="68" width="120" height="84" rx="12"/><rect class="bgpi-box" x="458" y="68" width="120" height="84" rx="12"/><rect class="bgpi-box" x="658" y="68" width="120" height="84" rx="12"/>
<text class="bgpi-text" x="118" y="103">origin AS</text><text class="bgpi-text" x="318" y="103">transit A</text><text class="bgpi-text" x="518" y="103">transit B</text><text class="bgpi-text" x="718" y="103">client ISP</text>
<text class="bgpi-small" x="118" y="130">route held</text><text class="bgpi-small" x="318" y="130">route held</text><text class="bgpi-small" x="518" y="130">route held</text><text class="bgpi-small" x="718" y="130">route held</text>
<text class="bgpi-x bgpi-a" x="118" y="58">×</text><text class="bgpi-x bgpi-b" x="318" y="58">×</text><text class="bgpi-x bgpi-c" x="518" y="58">×</text><text class="bgpi-x bgpi-d" x="718" y="58">×</text>
<text class="bgpi-small" x="430" y="198">Each AS updates its own route view after its neighbor withdraws.</text>
</svg>
<figcaption>A withdrawal is local first. The loss of reachability spreads as neighboring ASes remove their learned route.</figcaption>
</figure>

**Important limit:** this is a path-state animation, not a timer. It would be dishonest to assign one universal “BGP convergence time.” Timers, topology, local policy, alternatives, and failure detection all vary. In an incident, record the time each independent observer saw the route change and the time each probe saw data-plane failure. The gap between those two timelines is evidence.

## 2. Meta withdrew its DNS reachability after its backbone failed

The Meta outage on **October 4, 2021** is often described as “Facebook disappeared from the Internet.” That phrase captures the user's experience, but it hides two distinct failures. Meta's [engineering account published October 5](https://engineering.fb.com/2021/10/05/networking-traffic/outage-details/) says a routine maintenance command, intended to assess global backbone capacity, unintentionally disconnected its data centers from each other. A bug in the audit system failed to stop the command. Then Meta's DNS-serving facilities judged themselves unhealthy because they could not speak to the data centers and withdrew the BGP advertisements for their authoritative name servers.

![Meta backbone disconnection causing health-driven withdrawal of authoritative DNS routes](/imgs/blogs/bgp-incidents-hijacks-leaks-and-the-day-facebook-withdrew-itself-3.webp)

The cause chain matters more than the headline. The command was the trigger. The audit bug was a failed guardrail. The backbone disconnection made serving sites unable to communicate with the systems behind them. The DNS route withdrawal was an automatic, intended behavior under that health condition. The result was that the authoritative DNS servers could still be operational as machines while their addresses were unreachable from the wider Internet. A resolver could not discover where the application lived. Existing cached answers might still point at addresses, but the backbone failure could keep those destinations from working as well.

Meta's [same-day update](https://engineering.fb.com/2021/10/04/networking-traffic/outage/) states that the underlying cause was a faulty configuration change, not malicious activity. There is no evidence in Meta's report of a third-party AS stealing Meta's DNS prefixes. This was self-withdrawal after an internal failure, not an origin hijack. Route origin validation would not have kept an intentionally withdrawn route alive. Even a perfectly signed route authorization cannot tell the world to use a route the origin has stopped advertising.

The outage also attacked recovery. Meta says normal network access to data centers was unavailable, internal tools lost DNS, and out-of-band access was down. Engineers had to go onsite, where physical and system security slowed access to the relevant equipment. Those controls were there for good reasons. The incident shows that a recovery path that shares the failed network's dependency is not independent, however different its login procedure looks. In the postmortem, Meta also describes a cautious restoration because turning services on at once could create another failure as traffic returned. That is an operational consequence of a large withdrawal and reannouncement: the route returning is necessary but not sufficient for safe recovery.

An application team should not infer “DNS error equals DNS operator error” from this case. DNS was the first visible failure for many clients, but its reachability depended on the backbone. The discriminating checks are: can independent vantage points reach the authoritative server's IP address; are its prefixes visible in public route collectors; can Meta's own internal probes reach backends; and does the DNS health condition depend on the same backbone that the DNS route is meant to route around? Those checks separate serving-process health from network reachability.

**Transfer rule:** if an edge withdraws a route based on backend health, rehearse a failure in which the health check, backend, management channel, and DNS path all share one fault domain. The question is not whether withdrawal is good or bad in general. The question is whether the health predicate actually represents the service's ability to serve clients, and whether humans can still reach the control plane when the predicate turns false everywhere.

## 3. Rogers: an internal route filter disappears

On **July 8, 2022**, Rogers customers across Canada lost wireless and wireline services. The [Canadian Radio-television and Telecommunications Commission's independent assessment](https://publications.gc.ca/collections/collection_2024/crtc/BC92-130-1-2024-eng.pdf), published in 2024, identifies a specific trigger: a policy filter was removed from a distribution router configuration at **04:43 EDT**. The change allowed full BGP routing tables to be redistributed into Open Shortest Path First (OSPF), the internal routing protocol. The resulting flood exhausted CPU and memory on core routers. The assessment says core gateways began failing within two minutes, followed by nationwide loss of Rogers wireless and wireline services.

![Rogers filter removal and subsequent core router overload on July 8, 2022](/imgs/blogs/bgp-incidents-hijacks-leaks-and-the-day-facebook-withdrew-itself-4.webp)

This was not a global attacker announcing Rogers' addresses. The failure was inside Rogers' own control plane. BGP carries reachability between networks; OSPF distributes reachability within a network. Importing all external routes into an internal domain without a narrow policy can multiply the amount of state every participating router must process. The filter was the boundary. Removing it did not merely choose a less optimal path. It exposed core routers to a volume of route updates for which that part of the network was not prepared.

Rogers' [CEO statement on July 9](https://about.rogers.com/news-ideas/a-message-from-rogers-president-and-ceo/) said the disruption followed a maintenance update in the core network and acknowledged that some customers could not reach emergency services. The later CRTC report supplies the more detailed causal account. Keep those publication dates distinct from the event date. A contemporary customer-facing statement is useful for the symptom and public impact; the later investigation is stronger evidence for the route redistribution mechanism.

The assessment also reports that engineers lost access to parts of network management during the outage and that recovery was staged. Mobile registration had to be throttled to avoid a signaling storm as devices returned. That is the same second-order lesson visible in Meta's restoration: once a network has been absent, putting the route back can unleash queued demand, retries, registrations, and cold caches. The application team's normal retry policy may amplify the returning traffic. A network fix can therefore produce a new load problem if every client tries again immediately.

For this class of event, RPKI is not the primary safeguard. A Route Origin Authorization (ROA) can say which AS may originate a public prefix. It cannot set a CPU budget for an internal OSPF domain or prevent an authorized engineer from removing a redistribution filter. The relevant controls are configuration review that understands policy diffs, maximum-prefix and route-count limits at boundaries, staged rollout, overload protection on core control planes, and management access that survives the same internal failure. An application engineer may not configure any of those, but can recognize the symptom: multiple unrelated services on one provider fail together, public BGP visibility changes may be downstream of core collapse, and an application deployment rollback would be noise.

**Transfer rule:** treat redistribution boundaries like schema boundaries. A route accepted by one protocol is not automatically safe for another. Require a bounded set of prefixes and a tested maximum state change before changing the policy that crosses that boundary.

## 4. KT Korea: a local routing command escaped its region

On **October 25, 2021**, KT customers in South Korea experienced a nationwide network disruption. The most useful source is the [Korean Ministry of Science and ICT's investigation briefing, published October 29](https://www.korea.kr/briefing/policyBriefingView.do?newsId=156477990). The ministry says a worker entered an incorrect configuration command during a router replacement at KT's Busan facility. The resulting routing error propagated across the country. Its investigation rejected the early hypothesis of a distributed denial-of-service (DDoS) attack after examining IP patterns and traffic. A [government policy summary](https://eiec.kdi.re.kr/policy/materialView.do?num=219615&topic=) says the disruption lasted about **89 minutes**, from around **11:16** to **12:45** local time. These are source-reported times, not a timing property of BGP.

This case is useful because the first explanation was wrong. The ministry's account says the maintenance was performed during the day despite a plan for night work, without the planned supervision, and while the network remained connected. It also says KT lacked a mechanism to stop a regional mistake spreading nationally. Each condition changed the blast radius: the wrong command started the fault; live propagation carried it; missing containment enlarged it; the work schedule increased the number of users exposed. None of those facts proves an external prefix hijack.

Notice the diagnostic trap. A DDoS attack and a routing loop or route error can both produce high traffic or overloaded components. If a team sees traffic spikes and immediately labels the event an attack, it may start blocking sources while a bad route continues propagating. A more discriminating question is whether the suspicious traffic is the cause of overload or the result of packets and control-plane updates circling through changed routes. Correlate the precise route-change timestamp with interface counters, route-table diffs, and traffic direction. The ministry had the advantage of internal evidence; an application team outside KT would need to state uncertainty until the operator confirmed the mechanism.

This incident belongs next to Rogers because both were caused by an internal change, yet their precise failure modes differ. Rogers' documented mechanism is a removed redistribution filter that let a full BGP table flood OSPF and overload core routers. KT's published government account emphasizes an incorrect router command and nationwide propagation. We should not borrow Rogers' OSPF explanation and paste it onto KT. A shared category such as “routing configuration fault” is useful only if the distinct causal chains remain visible.

For an application on KT's network, the safe response is evidence collection and dependency failover, if a genuinely independent path exists. Before triggering a global application rollback, check whether unrelated destinations over the same access provider fail. Compare from a separate carrier. If a payment gateway, DNS resolver, and your origin all become unreachable only from KT, the common denominator is likely below your application. But “likely” is the right word until the route or operator data arrives.

**Transfer rule:** change controls must limit propagation, not merely catch typos. A correct review process asks what part of the network can receive the changed route, how many routers can consume it, and how to revert if the management path fails. An application team's analogous control is a regional canary with an independent rollback route. The analogy is operational, not a claim that an app deployment can repair an ISP.

## 5. Pakistan Telecom and YouTube: the more-specific route wins

The **February 24, 2008** YouTube incident is the cleanest way to see why an unauthorized origin matters. Pakistan Telecom originated `208.65.153.0/24`, a more-specific part of YouTube's `208.65.152.0/22` space, and that route escaped through its upstream provider, PCCW. A [RIPE NCC analysis of the observed hijack](https://www.ripe.net/documents/3520/MENOG3-dranse-youtube.pdf) records the announcement history, while a [RIPE Labs routing security explainer](https://labs.ripe.net/author/remy_de_boer/secure-internet-routing-with-rpki/) identifies Pakistan Telecom's AS17557 and the missing upstream filter. The locally intended censorship or null route became a global reachability problem because the advertisement crossed a provider boundary that should have stopped it.

The essential math is address containment, not speed. A /22 IPv4 block contains ${2}^{32-22} = 1024$ addresses. A /24 contains ${2}^{32-24} = 256$ addresses. Thus the leaked /24 covered **one quarter** of the legitimate /22 by address count. Those numbers are derived from prefix length; they do not mean exactly one quarter of YouTube users were affected. Address popularity, server placement, caches, and the set of networks that accepted the route determine user impact. For a destination inside that /24, a router with both prefixes installed follows the /24. The real origin can be announcing the broader /22 correctly and still lose traffic for those addresses.

There are two boundary failures here. The first is origination: AS17557 announced address space it was not authorized to originate globally. The second is export: a provider carried that announcement outward. The word “hijack” describes the false origin's effect, while “leak” describes the route escaping the intended scope. The distinction is not pedantic. Origin validation addresses the first boundary for networks that enforce it. Customer prefix filtering and export policy address the second. If either had been universally effective along the observed propagation path, the global effect could have been reduced. A single operator's control is not proof of global immunity.

The packet symptom can vary by location. A client whose ISP never accepted the more-specific route could continue using YouTube's legitimate path. A client behind an AS that accepted it could send packets toward Pakistan Telecom. A public route collector might see the false origin from some peers and not others. This is why an incident dashboard should preserve vantage point, source AS, and destination IP with every probe. “YouTube is down” is a user report. “Connections to addresses within `208.65.153.0/24` fail from these ASes after the new /24 announcement” is a routing hypothesis one can test.

The event also shows why changing a DNS record is not necessarily the right first response to a hijack. DNS names point at IP addresses. If the current address is in a hijacked /24, a fresh DNS answer containing the same address still follows the hijacked route. A failover to a genuinely different provider and prefix could help, but only if clients can resolve the new answer and the alternative path does not share the same routing fault. That introduces TTL, caching, certificate, and capacity considerations covered in [DNS in production](/blog/software-development/networking/dns-in-production-ttls-caching-layers-and-stale-answers). It is a contingency, not an instant delete button.

**Transfer rule:** ask your provider which exact customer prefixes it accepts and exports, and whether it validates origin authorization. A route object that exists in a database is not the same thing as a filter deployed at every neighbor. Verify observed announcements from independent collectors after a change. For the application team, maintain a list of critical IP prefixes and ASNs so an incident responder can recognize a suspicious more-specific instead of staring only at hostnames.

## 6. Route 53 and MyEtherWallet: a routing attack becomes a DNS attack

On **April 24, 2018**, an attacker-controlled routing path diverted traffic toward some Amazon Route 53 authoritative DNS addresses. Cloudflare, whose public resolver was affected in some locations, [published a contemporaneous technical analysis](https://blog.cloudflare.com/bgp-leaks-and-crypto-currencies/). It records announcements of several more-specific `205.251.*.0/24` prefixes from AS10297 beginning around **11:05 UTC**, with its own observed interval ending around **12:55 UTC**. Those /24s sat within Amazon's broader Route 53 space. Cloudflare says its resolver sites in a named set of cities received false DNS responses for `myetherwallet.com`, while others worked normally. [MyEtherWallet's June 4 follow-up](https://medium.com/@myetherwallet/a-message-to-our-community-a-response-to-the-dns-hack-of-april-24th-2018-26cfe491d31c) confirms that a BGP hijack of Amazon DNS infrastructure redirected some users to a phishing page.

![A more-specific BGP announcement diverting Route 53 queries and contaminating recursive DNS answers](/imgs/blogs/bgp-incidents-hijacks-leaks-and-the-day-facebook-withdrew-itself-5.webp)

Follow the two planes separately. First, an unauthorized BGP origin for Route 53 address space changed where some resolvers sent DNS packets. Second, an impostor server answered queries for the wallet domain with an attacker-chosen address. Third, a resolver that accepted and cached that DNS answer could give the bad address to clients even when those clients were not themselves located behind an AS that accepted the BGP hijack. Cloudflare explicitly notes this indirect effect. The attack's routing scope and its DNS answer scope were not identical.

That is a surprising property for an application engineer. A user can have a perfectly ordinary route to the final website address and still receive the wrong address because their recursive resolver was poisoned upstream. Conversely, a different resolver can return the legitimate answer from the same city. Comparing only traceroutes to the web origin misses the first hop of the attack chain. During a similar incident, compare authoritative answers from multiple locations and recursive answers from multiple resolvers. Record the queried server IP, response, DNSSEC validation status if applicable, and time. Do not conflate DNSSEC with BGP origin validation: one authenticates DNS data under its trust chain, while the other authenticates which AS may originate a prefix. Deployment and validation behavior determine whether either helps a given query.

Cloudflare's report says the impostor site presented a self-signed TLS certificate and that users would have to accept the certificate warning for the web attack to proceed through the browser. That is a critical boundary. A route or DNS hijack can steer a connection, but it does not by itself grant a valid certificate for the target hostname. A mobile client or backend that disables certificate verification turns steering into potential content compromise. A browser user who overrides the warning crosses the same boundary. When an incident response team sees certificate errors alongside strange DNS answers, suppressing the errors to restore “availability” is exactly the wrong move.

Avoid overspecifying what the evidence does not prove. Cloudflare's observations establish the prefixes it saw, the resolver sites it identified, and the incorrect answers it received. They do not give a global count of affected users or a universal duration for every resolver cache. MyEtherWallet confirms the phishing target and the broad mechanism; a claim about total stolen funds would need a separate primary ledger and careful attribution. We do not need an uncertain monetary total to understand the failure.

**Transfer rule:** protect the integrity boundary even when reachability is under attack. Keep hostname verification enabled, investigate certificate errors as security signals, and compare DNS answers across independent recursive and authoritative paths. For critical operations, plan how to stop sensitive transactions when routing or name integrity is uncertain rather than encouraging users to click through TLS warnings.

## 7. RPKI and ROAs: what they prove, and what they cannot

Resource Public Key Infrastructure (RPKI) gives address holders a cryptographic way to state which AS may originate a prefix. A Route Origin Authorization (ROA) is the signed object that carries that assertion. It contains an authorized origin AS, a prefix, and optionally a maximum prefix length. [RFC 6482, published in February 2012](https://www.rfc-editor.org/rfc/rfc6482.html), defines the ROA profile. A validator checks the certificate chain and turns valid ROAs into validated ROA payloads. Routers can then perform Route Origin Validation (ROV) on received BGP routes, as specified in [RFC 6811, published in January 2013](https://www.rfc-editor.org/rfc/rfc6811.html).

The key word is **origin**. ROV compares the route's originating AS and prefix to authorization data. It does not verify every AS along the path, prove that packets reach the stated destination, certify DNS data, or guarantee that an authorized origin is healthy. It also does not create or preserve an announcement. A valid route can disappear because its owner withdraws it, exactly as Meta's DNS routes did. A provider can overload its internal routers with legitimately originated routes, as in Rogers. Those failures lie outside ROV's promise.

Take a deliberately illustrative ROA for `203.0.113.0/24`, origin AS64500, with no `maxLength` value. RFC 6482 says that omitting `maxLength` authorizes only the exact prefix length. A route for that exact /24 from AS64500 can match. A route for a covered /25 from AS64500 does not match this ROA. A route for the /24 from AS64501 also does not match. These are examples using documentation addresses and private-use ASNs, not deployed Internet records. If the address holder actually intends to originate the /25, it must authorize that more-specific appropriately before deploying origin validation filters that could reject it.

Now suppose the same ROA explicitly sets `maxLength` to 26. It authorizes prefixes within the /24 through /26 from AS64500. That is operationally convenient for legitimate traffic engineering, but it widens the set of more-specific announcements that could still appear origin-valid if they claim the authorized origin. [RFC 9319, published in October 2022](https://www.rfc-editor.org/rfc/rfc9319.html), discusses the security and operational trade-offs of `maxLength`. The right value is the narrowest one consistent with actual announcement plans. Do not set it to the longest possible prefix simply to avoid future paperwork.

[RFC 6811](https://www.rfc-editor.org/rfc/rfc6811.html) gives a route one of three validation states. **Valid** means at least one validated payload matches both prefix constraints and origin AS. **Invalid** means at least one payload covers the prefix, but none matches the route. **NotFound** means no validated payload covers that prefix. NotFound is not the same as Invalid: the absence of a ROA does not cryptographically say that every possible origin is wrong. A common operator policy is to reject Invalid and handle Valid and NotFound under other routing rules, but the RFC defines validation state rather than a universal business policy for every deployment.

Consider the YouTube pattern as a counterfactual, not a claim about deployed controls in 2008. If the legitimate holder had a correct ROA covering the hijacked /24's parent prefix and authorizing only YouTube's origin, Pakistan Telecom's AS17557 announcement would be Invalid under ROV. A neighboring network enforcing rejection of Invalid routes would not install it. However, protection would depend on correct ROAs, a working validator, and enforcement along the relevant propagation paths. Networks that do not validate could still accept the route. A malformed ROA could also mark the legitimate more-specific Invalid. RPKI is a strong control for a specific class of unauthorized origin, not a universal BGP firewall.

The Route 53 incident shows another subtlety. The false /24s were more-specific than Amazon's broader announcements. A correct ROA for Amazon space and origin, combined with ROV that rejects Invalid, could make those unauthorized origin routes fail validation at enforcing networks. But even if one resolver's upstream rejected the false route, a different resolver might accept it and cache a bad DNS answer. The application client using that resolver could still receive poisoned data. Routing security and DNS integrity must be evaluated at the actual resolver path, not just at the client's ISP.

### A practical ROA rollout checklist

An application engineer who does not own an ASN should still know what to ask of the infrastructure provider. If your company does own prefixes, the network team needs an inventory of every origin AS and announced more-specific used for active service, failover, and DDoS mitigation. Compare that inventory with existing ROAs before enabling rejection of Invalid routes. A missing legitimate origin or a too-short `maxLength` can turn a resilience maneuver into a self-inflicted outage. Conversely, an excessively broad authorization loses some of the precision that makes origin validation valuable.

The rollout sequence is deliberately boring: document announced prefixes, create or correct ROAs, check observed route validation state from multiple independent perspectives, test failover announcements, then enable or expand filtering with rollback criteria. The exact RIR portal and router commands depend on the address holder and platform, so copying a generic ROA creation command into production would be unsafe. The principle is stable: authorization data and actual BGP advertisements must agree **before** an operator starts dropping Invalid routes.

When a prefix is Invalid in one looking glass, inspect the prefix, origin AS, and maximum length separately. The fault could be a genuine hijack, an unregistered failover origin, a deaggregation beyond `maxLength`, or stale validation data. The observation is a signal for investigation. Do not immediately “fix” it by broadening every ROA. That would remove the constraint designed to catch accidental or hostile more-specific announcements.

### MANRS deals with the operator boundary

Mutually Agreed Norms for Routing Security (MANRS) is an operator practice framework, not a new packet protocol. Its [network operator actions](https://manrs.org/netops/actions/) cover filtering incorrect routing information, preventing spoofed source addresses, maintaining operational coordination, and facilitating global validation. These actions address different failure points. Customer-prefix filtering helps stop an unauthorized announcement before it leaves a provider, even when downstream networks lack ROV. Coordination ensures there is a reachable person to contact when a route looks wrong. Anti-spoofing improves a different security property and should not be described as an anti-hijack mechanism.

The relevant question for an application team's vendor review is more specific than “Are you MANRS compliant?” Ask whether the provider filters customer announcements by authorized prefix and origin, how it limits more-specifics, whether it rejects RPKI Invalid routes, how quickly it can withdraw a bad announcement, and whether its escalation path works when its own data network is down. The answers can reveal a real boundary or only a policy document. The five incidents show why those details matter: a legitimate self-withdrawal, a missing export filter, and a poisoned DNS path demand different controls.

## 8. What an application engineer can actually measure

You probably cannot edit your transit provider's import policy at 03:00. You can avoid wasting the first hour on an application rollback. The response starts by separating name resolution, route visibility, packet delivery, TLS integrity, and application response. A single red synthetic check collapses all five into “down”; a useful check preserves where the failure first appeared.

![An application engineer's decision tree for DNS, route, packet, TLS, and application failures](/imgs/blogs/bgp-incidents-hijacks-leaks-and-the-day-facebook-withdrew-itself-6.webp)

Start with a fixed hostname and a fixed destination IP from a previously known-good answer. Resolve the hostname through your configured resolver and an independent resolver. Query the authoritative name server directly if you know its address and can reach it. Probe the fixed IP while preserving the original hostname for HTTP Host and TLS Server Name Indication. Compare results from multiple source networks. A success through one path and a failure through another is not a contradiction; it is often the strongest clue that the problem sits in routing or a regional resolver.

Here is a read-only command sketch for a real service. Replace the documentation values with the service's published hostname, known-good edge address, and authoritative name server. Run the independent resolver check only if your incident policy permits sending the queried name to that resolver. This is a procedure, not output from a production incident.

```bash
HOST=www.example.com
EDGE_IP=203.0.113.10
AUTH_IP=192.0.2.53

date -u +%FT%TZ
dig +time=2 +tries=1 "$HOST" A
dig +time=2 +tries=1 @"$AUTH_IP" "$HOST" A
dig +time=2 +tries=1 @1.1.1.1 "$HOST" A

curl --connect-timeout 3 --max-time 10 \
  --resolve "$HOST:443:$EDGE_IP" \
  -o /dev/null -sS \
  -w 'remote_ip=%{remote_ip} dns=%{time_namelookup} connect=%{time_connect} tls=%{time_appconnect} status=%{http_code}\n' \
  "https://$HOST/"
```

The `--resolve` probe pins the destination IP but still uses the hostname for the HTTPS request. It is useful for asking whether the current DNS answer is the main difference. It does **not** prove the BGP path is safe. If the pinned IP's prefix is hijacked, packets can still follow the wrong origin. It also does not prove the server is healthy if a CDN uses different backends for different client networks. Record the actual `remote_ip`, source network, and timestamp with the result.

Interpret timing fields carefully. A `curl` DNS time near zero with `--resolve` is expected because the command supplied the address. A failed TCP connect with no HTTP status says the request did not reach the HTTP layer; it does not alone distinguish routing from a firewall or dead listener. A TLS certificate error should remain a hard failure and be investigated separately from reachability. An HTTP error status means some server answered, which is very different from no packets arriving. This is the same layer-by-layer discipline used in [the curl path post](/blog/software-development/networking/what-actually-happens-when-you-curl-a-url).

For route visibility, inspect the exact prefix and origin in more than one public collector. [RIPE RIS documentation](https://ris.ripe.net/docs/) explains that collectors receive BGP data from their own peers, and its [route collector guide](https://ris.ripe.net/docs/route-collectors/) describes why each view is partial. [RIPEstat's BGP State endpoint](https://stat.ripe.net/docs/data-api/api-endpoints/bgp-state) provides a route view through an API. During an incident, save the JSON response or collector screenshot with a UTC timestamp. Look for a new origin AS, a more-specific covering the failing address, a withdrawal, or a changed path. Then ask whether the change was visible at collectors close to affected client networks. “A collector saw it” supports a hypothesis. It does not certify the route taken by one client packet.

The packet path is similarly easy to overclaim. `traceroute` or `mtr` can show that traffic stops or detours, but routers may rate-limit probe replies, hide hops, or choose a different load-balanced path for probe packets. A silent intermediate hop with successful destination traffic is not loss at that hop. Use traceroute as directional evidence, paired with the route collector and end-to-end probe. For production changes, leave these commands read-only. Do not delete routes or flush DNS caches on a host whose current network state is your evidence.

### The incident matrix

During response, write one row per vantage point, not one row per dashboard. Include time in UTC, source ASN or provider, resolver used, DNS answer, destination IP, TCP result, TLS result, HTTP result, and any observed route origin. This is an evidence schema; it intentionally contains no invented measurements. It forces the team to keep contradictory observations visible instead of averaging them into “the site was down.”

| Observation | More likely boundary | Next discriminating check |
| --- | --- | --- |
Authoritative IP unreachable and its prefix withdrawn | Origin or provider reachability | Origin health condition and route history |
| Public collectors see an unauthorized more-specific | Interdomain announcement | Source-AS probes, ROA state, provider escalation |
| Only some recursive resolvers return a strange answer | Resolver path or cache | Query authoritative servers from multiple networks, preserve TLS errors |
| Fixed-IP HTTPS works but ordinary hostname fails | DNS resolution or answer selection | Compare recursive and authoritative answers with TTL |
| DNS and TCP work but TLS hostname verification fails | Endpoint identity | Inspect certificate chain and actual remote address |
| Many unrelated destinations fail through one access provider | Access or provider core | Test through independent carrier, page provider |

This table is a triage aid, not a deterministic classifier. A single event can cross boundaries. The Route 53 attack began in BGP and produced bad DNS data. Meta's backbone failure produced both unreachable application paths and DNS self-withdrawal. The benefit of the matrix is that it tells you what to measure next, not that it automatically writes the postmortem.

### Two worked investigations

**Investigation A: a missing route.** Assume, as a hypothetical example, that your authoritative server lives at `203.0.113.10` and your published prefix is `203.0.113.0/24`. From two independent carriers, `dig @203.0.113.10` times out. Public collectors that previously saw the /24 now show its withdrawal, and your internal DNS process health check remains green. Those observations make “the daemon crashed” weaker than “the route to the daemon disappeared.” If your edge withdraws BGP based on backend health, inspect that predicate and the dependency graph. Meta's 2021 case demonstrates that a healthy DNS process can be unreachable after a self-withdrawal, but the hypothetical observations do not prove the same trigger without your own control-plane logs.

**Investigation B: a bad origin.** Assume a different hypothetical destination at `203.0.113.10`. One collector shows an unexpected `203.0.113.0/24` originated by AS64501, while the legitimate operator normally originates a covering `203.0.112.0/23` from AS64500. Affected source networks fail to connect, while an unaffected source succeeds. The /24 is more specific than the /23, so it can attract packets in networks that accept it. You still need to confirm whether the new origin is unauthorized, whether the route was actually installed at affected networks, and whether any alternate path exists. The right escalation includes exact prefix, origin AS, first observed UTC time, and affected vantage points. “BGP is broken” gives a provider much less to act on.

In both investigations, choose a remedy that matches the evidence. An application rollback does not restore a withdrawn prefix. A DNS update to an IP in the same hijacked block may change nothing. A failover to a separate provider might restore some clients, but it can create a traffic surge and a new capacity bottleneck. If the alternative shares a transit dependency, apparent provider diversity may collapse under the same routing event. Test the failover path from several source ASes before treating it as independent.

## 9. Build a service that survives a routing incident

The first design action is to map dependencies by **prefix and provider**, not only by cloud region name. A multi-region application can still depend on one authoritative DNS operator, one anycast prefix, one transit relationship, or one corporate management network. A second cloud account on the same provider edge does not necessarily give an independent interdomain path. The question for a resilience review is: if a critical prefix is withdrawn or falsely originated, which user journeys and recovery tools remain reachable?

Maintain a small routing inventory for critical public services: hostname, authoritative DNS provider, DNS server addresses and prefixes, application edge prefixes, origin ASNs, transit or CDN provider, certificate ownership, and an escalation contact. Keep a copy available outside the network whose outage it describes. For an application team, this inventory is not a second routing database to keep perfectly synchronized by hand; it is the minimum context required to recognize a new origin or a common prefix across otherwise unrelated failing services. Reconcile it with the network team when a CDN, DNS provider, or address plan changes.

Make probes independent in the ways the incident can break. A monitor that uses the same corporate DNS resolver, egress provider, and cloud region as the application is excellent for detecting its own failure but weak for localizing it. Place read-only probes behind at least two source networks with different upstreams when the service's impact warrants it. Record DNS result, TCP connect result, TLS verification, HTTP response, and selected address. A dashboard can aggregate these fields later; collection should retain the raw dimensions.

Separate user-serving failover from management access. Meta and Rogers both describe recovery complications when their normal control paths were affected. A management channel is independent only if its power, identity provider, DNS, network transport, and physical access plan survive the failure being rehearsed. The answer may be a separate carrier, a secure console path, a documented onsite procedure, or a combination. The application team's role is to know whether it can still change configuration and communicate when the primary service network is gone.

Treat retries as a restoration hazard. When a route returns, clients that spent the outage retrying may all reconnect. A service with unlimited retries can turn a successful network recovery into a server overload. [Timeouts, retries, and backoff](/blog/software-development/site-reliability-engineering/timeouts-retries-and-backoff-done-right) and [cascading failures and bulkheads](/blog/software-development/system-design/cascading-failures-circuit-breakers-and-bulkheads) cover the application policy. The routing lesson is the trigger: restoring reachability changes offered load sharply. Confirm that the application can shed load and that clients use bounded, jittered retries.

Do not mistake an alternate DNS record for full path diversity. The alternate answer has to be reachable from affected source networks, supported by valid TLS for the hostname, able to take the load, and independent of the failed DNS path enough for users to learn about it. If the resolver cannot reach authoritative servers, a fresh failover record may not be seen until caches expire or resolution recovers. [Service discovery and load balancing](/blog/software-development/microservices/service-discovery-and-load-balancing) discusses how discovery staleness enters the application boundary. This post's boundary is the route that makes the discovery system reachable in the first place.

Finally, arrange a provider escalation template in advance. Include the failing destination IP and prefix, source ASN and location, UTC start time, traceroute as supporting data, DNS answer and resolver, observed BGP origin and more-specific, RPKI state if checked, and the control-plane change owner if it is your own network. The format reduces the time lost to vague tickets. It also prevents a common incident mistake: trying to solve an external route leak by restarting an otherwise healthy application.

### Choose mitigations by the boundary they can change

An incident often produces a tempting list of actions: lower DNS TTL, switch CDN, announce a more-specific, block an ASN, disable health checks, or force traffic through a different region. Each action changes a different boundary. Lowering TTL affects how long cooperating resolvers reuse an old answer after they learn a new one. It cannot force a resolver to reach an authoritative server whose prefix is withdrawn. Switching CDN helps only when the alternate CDN uses an independently reachable DNS and address path and can accept the offered traffic. Announcing a more-specific can regain traffic from a false origin in some networks, but it can also be filtered, invalidate your ROA, fragment traffic, or start a new routing contest. The last option belongs with the network operator, not an on-call application engineer improvising at a terminal.

Disabling a health-driven withdrawal can restore a BGP advertisement while leaving the actual backend unreachable. In the Meta pattern, that might make DNS server addresses visible again without fixing the disconnected backbone behind them. A client could then receive an application address and still fail on the next step. Such an action is useful only if the health predicate is known to be falsely negative and there is a separate safe serving path. Similarly, blocking a suspicious source ASN at an application firewall does not stop a route announcement; the wrong-origin network may never send an HTTP request to your edge at all. Measure which layer the proposed change touches before applying it.

If the incident is a verified unauthorized origin, the operational escalation is to the address holder, its upstream providers, and any network accepting or propagating the route. Provide prefix, origin AS, timestamps, and collector evidence. If it is a self-withdrawal, focus on the origin's health condition and the dependency that made it withdraw. If it is internal route redistribution, prioritize rollback of that specific policy change and protect the control plane from overload. The application team can meanwhile reduce retry pressure and steer only traffic for which an independent alternative has been tested.

This is where [load balancing from L4 to L7](/blog/software-development/system-design/load-balancing-from-l4-to-l7) reaches its boundary. An L7 load balancer can pick a healthy backend only after the packet reaches it. It cannot choose a backend for packets diverted to a false origin or dropped before its network. The same separation applies to [observability by design](/blog/software-development/system-design/observability-metrics-logs-traces-by-design): traces begin where the application can observe a request. A missing trace may be evidence that the failure occurred before the first instrumented component, but only an external probe and route evidence can place it.

## 10. Run it yourself

### Question

Can a healthy destination stop receiving packets solely because the source's forwarding table no longer has a usable route to its address? The lab tests that necessary part of the Meta-style reachability story. It does **not** run BGP, reproduce Meta's outage, model Internet convergence, or test RPKI. A local Linux blackhole route stands in for the data-plane effect of a missing usable interdomain path. That scope is what makes the experiment short, safe, and falsifiable.

### Preconditions

Use a Linux host or a privileged Linux VM with `iproute2`, `ping`, root or `CAP_NET_ADMIN`, and the canonical `netlab` namespaces `c` and `s` from [the series setup post](/blog/software-development/networking/what-actually-happens-when-you-curl-a-url). The client has `c0=10.77.0.1/30`; the server has `s0=10.77.0.2/30`. On macOS, run these commands inside the Linux environment described by that setup. Do not run the route mutation on a production interface or host namespace. The test uses `ping` as a packet-delivery probe, not as proof of HTTP health. If the destination blocks ICMP or the namespaces differ from the canonical setup, fix the lab setup first.

The following preflight is read-only. It checks privileges, both namespace names, both interface addresses, the current route, and qdisc state. It also detects a stale treatment from an interrupted prior run. Run the commands in a Linux shell:

```bash
set -euo pipefail
test "$(id -u)" -eq 0 || { echo "Run as root in the lab VM"; exit 1; }
command -v ip >/dev/null
command -v ping >/dev/null
ip netns list | grep -E '^c([[:space:]]|$)'
ip netns list | grep -E '^s([[:space:]]|$)'
ip -n c -br addr show dev c0
ip -n s -br addr show dev s0
ip -n c route show
ip -n c qdisc show dev c0
ip netns exec c ip route get 10.77.0.2
if ip -n c route show 10.77.0.2/32 | grep -q '^blackhole '; then
  echo "Stale lab blackhole route exists; remove it before baseline"
  exit 1
fi
```

Read the `c0` and `s0` address lines for the canonical `10.77.0.1/30` and `10.77.0.2/30` addresses. Read `ip route get` for a route using `c0` toward the server address. The exact spacing and neighbor details depend on the installed `iproute2` release; the expected qualitative state is a connected route to `10.77.0.2`, no `blackhole 10.77.0.2/32`, and no special qdisc treatment that you forgot from another experiment. The lab does not alter qdisc state.

### Baseline

```bash
ip netns exec c ip route get 10.77.0.2
ip netns exec c ping -n -c 3 -W 1 10.77.0.2
```

Read the route's output interface and the `packets transmitted`, `received`, and `packet loss` fields in `ping`'s summary. In a working local veth pair, expect **3 transmitted, 2–3 received, and 0–33% loss**; most quiet lab runs receive all three. That is an expected lab range, not a production measurement. A virtualized host can lose a probe because of scheduler delay or an unfinished namespace setup. If the baseline loses all packets, the treatment cannot demonstrate a new failure and you should stop.

The destination namespace can remain healthy throughout. Confirm its interface is still up:

```bash
ip -n s -br link show dev s0
ip -n s -br addr show dev s0
```

These are destination-state checks, not application checks. If you also have the series `netserver` running at `10.77.0.2:8080`, a baseline `netclient --url http://10.77.0.2:8080/echo --requests 1 --concurrency 1 --json` can supply an application-layer observation. The route experiment does not require that process and does not start or stop it.

### Apply one change

Add exactly one /32 blackhole route **inside namespace `c`**. A host route is more specific than the connected /30, so the client chooses it for the one server address. The server interface, IP address, process, and namespace are unchanged. This mimics the forwarding consequence of “no usable path” while remaining confined to the lab:

```bash
ip -n c route add blackhole 10.77.0.2/32
ip -n c route show 10.77.0.2/32
```

Read the route line. It should begin `blackhole 10.77.0.2` or the equivalent prefix form for your `iproute2` version. This is one controlled mutation of the client's forwarding table. If `route add` says “File exists,” do not keep stacking changes. Inspect and remove only the stale lab route after confirming it is the one from this experiment.

### Compare

```bash
ip -n c route show 10.77.0.2/32
ip netns exec c ip route get 10.77.0.2 || true
ip netns exec c ping -n -c 3 -W 1 10.77.0.2 || true
ip -n s -br addr show dev s0
```

Read the route line, the `route get` error or route type, the `ping` summary, and the server address line. The expected result is that the /32 blackhole remains installed, `route get` does not yield a usable `c0` forwarding path, and **0 of 3** ICMP echo replies return. Depending on kernel and `iproute2`, `route get` may print a local error instead of a normal next-hop line; that difference does not change the claim. The server should still show its address. If the probe unexpectedly succeeds, check whether it actually ran inside `c`, whether the /32 route was installed there, and whether the destination was `10.77.0.2`. Do not explain away a successful treatment as scheduler noise.

The causal interpretation is narrow. We changed only the route selected by the client and saw packet delivery change while the server's interface stayed present. This is why an application origin can show no incoming requests even though its process is alive. In Meta's real event, a health-driven BGP withdrawal removed reachability from outside networks after a backbone failure. In this lab, a local blackhole produces a similar forwarding symptom without recreating the interdomain mechanism.

### Reset

```bash
ip -n c route del blackhole 10.77.0.2/32
ip -n c route show 10.77.0.2/32
ip netns exec c ip route get 10.77.0.2
ip netns exec c ping -n -c 3 -W 1 10.77.0.2
```

The first command removes only the route this experiment added. Expect no /32 blackhole route afterward and **2–3 of 3** replies again in the quiet local lab. If the final probe fails, do not delete the namespaces as a reflex. Inspect the baseline interface and route checks, then look for another experiment's impairment. This post does not change `tc`, `iptables`, `nft`, sysctls, or process state.

On a production host, the safe analogue is read-only: `ip route get DESTINATION_IP`, a bounded application probe, and a route-collector query for the public prefix. Do not install a blackhole route on a production namespace to “verify” an outage. Public route collectors also show only the views of their peers, so keep the end-to-end probe beside the announcement record. The test and the real diagnostic ask the same essential question: is the service absent because its process failed, or because packets lost a usable path before they reached it?

## Key takeaways

- A route hijack changes who claims a prefix. A leak moves an announcement outside intended policy. A self-withdrawal removes the rightful origin's own route. An internal redistribution error can collapse a network without an unauthorized Internet origin.
- The five cases map to different first boundaries: Pakistan Telecom and Route 53 involved false more-specific origins; Meta withdrew DNS routes after a backbone fault; Rogers flooded its core after removing a filter; KT let an incorrect local routing change spread nationally.
- RPKI ROAs and ROV can reject covered routes with unauthorized origins at enforcing networks. They do not validate a full AS path, keep withdrawn routes alive, protect internal OSPF, or replace TLS and DNS integrity.
- Application teams can locate the failing layer by preserving source network, resolver, destination IP, TLS result, and route observation together. A healthy process with no incoming requests is compatible with a routing failure.
- Recovery itself creates load. Confirm that failover paths are genuinely independent and that client retries are bounded before turning an unavailable route back into a reachable one.

## Further reading

- [RFC 7908: route-leak problem definition](https://www.rfc-editor.org/rfc/rfc7908.html), the policy definition behind the word “leak.”
- [RFC 6482: ROA profile](https://www.rfc-editor.org/rfc/rfc6482.html) and [RFC 6811: prefix origin validation](https://www.rfc-editor.org/rfc/rfc6811.html), the precise authorization and validation semantics.
- [MANRS network operator actions](https://manrs.org/netops/actions/), practical routing hygiene across filters, source validation, and coordination.
- [RIPE RIS documentation](https://ris.ripe.net/docs/), for interpreting a public collector's partial view.
- [The senior engineer's network mental model](/blog/software-development/networking/the-senior-engineers-network-mental-model), the series capstone that puts routing, resolution, transport, and application symptoms on one diagnostic map.
