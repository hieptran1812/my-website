---
title: "Kubernetes for Microservices: The Essentials You Actually Need"
date: "2026-06-15"
publishDate: "2026-06-15"
description: "Not a Kubernetes course — the handful of objects and the one mental model that map directly to running a fleet of services, plus the requests-and-limits story behind most of your mystery latency."
tags:
  [
    "microservices",
    "kubernetes",
    "containers",
    "deployment",
    "autoscaling",
    "resource-limits",
    "infrastructure",
    "distributed-systems",
    "software-architecture",
    "backend",
  ]
category: "software-development"
subcategory: "Microservices"
author: "Hiep Tran"
featured: true
readTime: 51
image: "/imgs/blogs/kubernetes-for-microservices-the-essentials-1.webp"
---

ShopFast's order service had been running happily on Kubernetes for six months when, on the morning of a flash sale, its p99 latency tripled from 45ms to 140ms — but only on some requests, seemingly at random, and only when traffic was high. The dashboards were maddening: CPU was at 55%, well under any alarm; memory looked fine; the database was fast; no errors, no restarts, nothing in the logs. The on-call engineer spent two hours chasing ghosts in the application code before someone with grey hair in their beard typed one command — `kubectl top pods` followed by a look at the pod's resource spec — and said four words that explained everything: "you're being CPU throttled." The order service had a CPU *limit* of `500m` (half a core). Under load it wanted a full core, so the Linux kernel's CFS scheduler was pausing the container for tens of milliseconds at a time, every 100ms, to keep it under its quota. The "mystery" latency was the kernel doing exactly what the YAML told it to.

This is the thing about Kubernetes for microservices: the platform will do *precisely* what you declare, mechanically, across your whole fleet, whether or not you understood what you declared. Get the declaration right and Kubernetes is the tireless operations team you could never afford to hire — it schedules your services onto machines, restarts them when they crash, scales them up when traffic spikes and back down when it ebbs, rolls out new versions without dropping a request, gives every service a stable address, and load-balances across copies. Get it wrong — a memory limit a hair too low, a liveness probe that checks the database, a missing resource request — and the same automation becomes the most efficient outage-amplifier you have ever built, all forty pods making the same mistake at machine speed.

This post is deliberately *not* a Kubernetes course. There are a dozen of those, most of them 600 pages long, and you will not need most of what they contain to run microservices. What you need is a small set of objects, one genuinely important mental model, and an honest understanding of where the sharp edges are. By the end of this post you will be able to take ShopFast's order service from a container image to a running, scaling, self-healing fleet: a `Deployment` of N replicas behind a `Service`, reachable through an `Ingress`, configured by a `ConfigMap` and a `Secret`, autoscaled by an `HPA`, with `requests` and `limits` set so it neither hogs a node nor gets throttled into the latency you just read about. You will be able to read a rolling update as it happens, debug an `OOMKilled` pod, and — most important for a senior — decide honestly whether your team should be on Kubernetes at all, because for a great many teams the answer is no. The figure below is the whole post in one frame: the stack of objects a single service maps to, top to bottom.

![A vertical stack showing the Kubernetes objects one microservice maps to from Ingress at the top down through Service Deployment HPA Pod ConfigMap and container](/imgs/blogs/kubernetes-for-microservices-the-essentials-1.webp)

This post sits in Track 6, the deployment and infrastructure track. It builds directly on [containerizing microservices](/blog/software-development/microservices/containerizing-microservices-docker-best-practices) — Kubernetes runs containers, so everything there is the prerequisite — and it is where the [health checks](/blog/software-development/microservices/health-checks-readiness-liveness-and-self-healing) and [service discovery and load balancing](/blog/software-development/microservices/service-discovery-and-load-balancing) posts finally get their platform. It is *not* the [service mesh](/blog/software-development/microservices/service-mesh-istio-linkerd-when-you-need-one) post — the mesh is the next layer up, the one you add when plain Kubernetes networking and the resilience patterns baked into your code stop being enough, and we will mark exactly where that line is and hand off. It also sets up [deployment strategies](/blog/software-development/microservices/deployment-strategies-blue-green-canary-feature-flags) (blue-green, canary) and [configuration and secrets management](/blog/software-development/microservices/configuration-and-secrets-management), both of which take the raw objects here and use them properly.

## Why Kubernetes for microservices at all

Start from the problem, not the tool. You have decided — wisely or not, that is the [first post in this series](/blog/software-development/microservices/what-are-microservices-and-when-not-to-use-them)'s question — to run a fleet of services. Maybe twelve of them today, maybe two hundred eventually. Every single one of those services, regardless of what it does, needs the same boring operational things, and it needs them continuously, forever, at 3am on a holiday:

It needs to be **placed on a machine** that has room for it. It needs to be **restarted** when it crashes, because processes crash. It needs **more copies** when traffic rises and **fewer** when it falls, because you don't want to pay for forty idle instances at 4am or fall over at noon. It needs a **stable address** so other services can find it even though its individual instances come and go and live at ever-changing IPs. It needs **traffic load-balanced** across its copies so no single one is hammered. It needs a way to **roll out a new version** without taking the old one down first and dropping every in-flight request. It needs its **configuration and secrets** injected without baking them into the image. And it needs all of this to happen *automatically*, because if a human has to do it then a human has to be awake to do it, and you cannot staff a 24/7 NOC to babysit two hundred services.

Call that whole bundle the **operational tax**. Every service pays it. Before container orchestration, teams paid it by hand — Chef and Puppet recipes to place processes, custom shell scripts and `systemd` units to restart them, an Nginx config someone edited to add a backend, a Capistrano deploy script that took the old version down for ninety seconds, a runbook that said "if QPS exceeds X, SSH to the box and start two more workers." This worked, sort of, for a handful of services and a heroic ops team, and it fell apart the moment you had dozens of services changing weekly. The figure below contrasts the two worlds.

![A before and after comparison showing manual host selection crash restarts and fragile deploy scripts on the left versus the Kubernetes scheduler self-healing and built-in rolling updates on the right](/imgs/blogs/kubernetes-for-microservices-the-essentials-2.webp)

Kubernetes is, at its core, a system that **automates the operational tax**. You describe what you want — "I want six copies of the order service, version 2.3.1, each guaranteed half a core and 512MB of memory, reachable at this address, and I want it scaled up automatically when CPU passes 60%" — and the platform makes it so and *keeps* it so. That last clause is the whole game. It does not just set it up once; it watches, continuously, and corrects any drift. A pod dies? It makes a new one. A node fails and takes ten pods with it? It reschedules them onto healthy nodes. Traffic spikes? It adds copies. This is why Kubernetes is worth its considerable weight for a service fleet specifically: the operational tax that was killing your ops team is precisely the thing Kubernetes was built to automate, and it gets *more* valuable, not less, as the number of services grows. (The flip side — that it is heavy, complex, and frequently overkill — is the honest "when not to" discussion we will have near the end. Hold that thought.)

## The one idea that matters most: reconciliation

If you internalize exactly one thing from this post, internalize this. Almost everyone teaches Kubernetes as a catalogue of objects — here is a Pod, here is a Service, here is an Ingress, memorize them. That is backwards. The objects make sense only once you understand the *machine* they live inside, and that machine is a **reconciliation control loop**.

Here is the idea, built up from scratch. In most software you are used to *imperative* commands: "create three pods" is an instruction the system executes once, and then it's done. Kubernetes is **declarative**: you do not tell it to *do* anything. You tell it what you *want the world to look like* — the **desired state** — and you store that desire in the cluster. Then a set of background processes called **controllers** run an endless loop. Each controller wakes up, looks at the desired state you declared, looks at the **actual state** of the world right now, computes the difference, and takes whatever action closes the gap. Then it does it again. And again. Forever.

That's it. That is the entire soul of Kubernetes. You declare `replicas: 6`. The Deployment controller (well, the ReplicaSet controller it spawns) wakes up, counts the running pods, sees 5, notes the gap is "one short," and creates one pod. Next loop it counts 6, sees no gap, does nothing. A node dies and takes two pods with it; next loop the controller counts 4, sees the gap is "two short," and creates two more pods on healthy nodes. You never told it to "restart the dead pods" — you only ever told it "I want 6," and the loop relentlessly drives reality toward that number. The figure shows the loop.

![A graph of the reconciliation control loop where desired state and actual state both feed a controller that computes a diff creates a pod to close the gap and re-observes through the watch API](/imgs/blogs/kubernetes-for-microservices-the-essentials-3.webp)

Once this clicks, everything else in Kubernetes stops being arbitrary trivia and becomes obvious. Self-healing? Not a feature — just the reconciliation loop noticing actual ≠ desired and acting. Rolling updates? The Deployment controller changing the desired state in steps and letting reconciliation chase it. Autoscaling? A controller that *adjusts* the desired replica count based on a metric, after which ordinary reconciliation makes it real. There is no special "restart" subsystem, no separate "scaling" engine — there is one pattern, applied by dozens of controllers, each owning one kind of object. When you `kubectl apply` a YAML file, you are not commanding the cluster to do something; you are *editing the desired state*, and the controllers take it from there.

This reframing also tells you how to *operate* Kubernetes correctly. Never `kubectl edit` a live pod to fix it by hand and walk away — the controller will reconcile your manual change right back out, because the *declared* desired state still says otherwise. Always change the desired state (the YAML in your Git repo, ideally), apply it, and let reconciliation do the work. This is the entire foundation of GitOps: your Git repository *is* the desired state, and a controller continuously reconciles the cluster to match the repo. Fight the loop and you lose; work with the loop and it is the most reliable colleague you have ever had.

### Why declarative beats imperative for a fleet

A junior reasonably asks: why is this better than just running scripts? Because reconciliation is **self-correcting and idempotent**, and those two properties are exactly what you need when reality keeps changing underneath you. An imperative script that says "create three pods" is correct only at the instant it runs; thirty seconds later a pod has crashed and your three are two, and the script has no idea — it already finished. A declarative `replicas: 3` is correct *continuously*, because the controller never stops checking. You can run `kubectl apply` ten times in a row and the outcome is identical — applying a state you're already in is a no-op. That idempotency is what makes Kubernetes safe to drive from automation, from CI/CD, from a recovering control plane after an outage. The desired state is the source of truth, and truth that corrects itself is worth a great deal at 3am.

### What actually happens when you run `kubectl apply`

It's worth tracing the path of a single `kubectl apply -f deployment.yaml`, because it demystifies the whole system and explains why operating Kubernetes the declarative way is the only sane way. Your YAML lands first at the **API server**, the single front door to the cluster. The API server validates the object, runs it through admission control (which can mutate it — injecting defaults, sidecars, or rejecting it against policy), and then writes the desired state into **etcd**, the cluster's consistent key-value store. That write is the *only* thing your `apply` actually did. It did not create a pod. It did not contact a node. It edited a record in a database that says "the world should look like this."

Everything after that is controllers reacting. The Deployment controller is *watching* the API server for changes to Deployment objects; it sees the new desired state, creates or updates a ReplicaSet to match, and writes *that* desired state back. The ReplicaSet controller is watching for ReplicaSet changes; it sees it needs six pods, counts the running ones, and creates Pod objects for the difference — again, just writing desired Pod records, not placing anything. The **scheduler** is watching for unscheduled Pods; it finds each one a node that fits its requests (the bin-packing step) and writes the node assignment back. Finally the **kubelet** on each assigned node is watching for pods bound to *it*; it pulls the image, starts the container, and reports status back up. Every arrow in that chain is the same pattern — watch the API server, compare desired to actual, write the next layer's desired state — which is why the whole thing is just reconciliation, all the way down, with etcd as the single source of truth and the API server as the only writer to it.

This architecture is also why Kubernetes survives its own control plane failing. If the API server goes down, your *running* pods keep running — the kubelets already know what to run. You just can't make *changes* until it's back, because there's nowhere to write new desired state. And it's why "edit a live object by hand" loses: your manual change is a write to etcd that the owning controller immediately reconciles against the *higher-level* desired state still sitting above it. The robust move is always to change the highest-level declaration (the Deployment YAML in Git) and let the cascade flow down.

## The essential objects, mapped to one service

Now that you have the loop, the objects are easy, because each one is just "a kind of desired state with a controller behind it." We will walk the full set you need to run ShopFast's order service, from the unit of execution up to the front door. The figure below shows how they relate: a Deployment owns a ReplicaSet that owns the Pods, and a Service selects those same Pods by label.

![A graph showing a Deployment owning a ReplicaSet that owns three Pods with distinct IPs while a Service selects all three pods by label for load balancing](/imgs/blogs/kubernetes-for-microservices-the-essentials-4.webp)

### Pod — the unit of scheduling

A **Pod** is the smallest thing Kubernetes schedules and runs. It is one or more containers that share a network namespace (so they share an IP and can talk over `localhost`) and can share storage volumes. For a microservice, the common shape is **one Pod, one main container** — your order service. You'll sometimes add **sidecar** containers in the same Pod (a log shipper, a metrics exporter, or, the big one, a service-mesh proxy — that's the [service mesh post](/blog/software-development/microservices/service-mesh-istio-linkerd-when-you-need-one)), but the mental default is one app per Pod.

The crucial thing about a Pod: **it is mortal and disposable.** A Pod is never restarted in place when it dies; it is *replaced* by a brand-new Pod with a new name and a new IP. You almost never create a Pod directly. You let a higher-level controller create and manage Pods for you, because a bare Pod has no one watching it — if it dies, it's just gone. Which brings us to the Deployment.

### Deployment — replicas and rolling updates

A **Deployment** is the object you actually write for a stateless service. It declares: which container image to run, how many replicas you want, and how to roll out changes. Under the hood it manages a **ReplicaSet** (the controller that keeps exactly N pods alive), but you rarely touch the ReplicaSet directly — the Deployment is the right altitude. Here is ShopFast's order service Deployment, with the things that actually matter in production: probes, resource requests and limits, and a rolling-update strategy.

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: order-service
  namespace: shopfast
  labels:
    app: order-service
spec:
  replicas: 6
  selector:
    matchLabels:
      app: order-service
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 2          # add up to 2 extra pods during a rollout
      maxUnavailable: 0    # never drop below the desired count
  template:
    metadata:
      labels:
        app: order-service        # MUST match selector.matchLabels
    spec:
      terminationGracePeriodSeconds: 30
      containers:
        - name: order-service
          image: registry.shopfast.internal/order-service:2.3.1
          ports:
            - containerPort: 8080
          envFrom:
            - configMapRef:
                name: order-service-config
          env:
            - name: DB_PASSWORD
              valueFrom:
                secretKeyRef:
                  name: order-service-secrets
                  key: db-password
          resources:
            requests:
              cpu: "500m"        # guaranteed 0.5 core
              memory: "512Mi"    # guaranteed 512 MiB
            limits:
              cpu: "1000m"       # may burst to 1 core
              memory: "512Mi"    # hard ceiling; cross it and you are OOMKilled
          readinessProbe:
            httpGet:
              path: /readyz
              port: 8080
            initialDelaySeconds: 5
            periodSeconds: 5
            failureThreshold: 3
          livenessProbe:
            httpGet:
              path: /livez
              port: 8080
            initialDelaySeconds: 15
            periodSeconds: 10
            failureThreshold: 3
          startupProbe:
            httpGet:
              path: /livez
              port: 8080
            failureThreshold: 30
            periodSeconds: 2     # allow up to 60s to start before liveness kicks in
```

Read that top to bottom and you have most of what a production service needs. The `selector` and the pod template `labels` must match — this is how the ReplicaSet knows which pods are "its" pods, and it's the single most common copy-paste bug. `maxUnavailable: 0` with `maxSurge: 2` means a rollout always adds new pods *before* removing old ones, so capacity never dips below your desired six — we'll trace that in detail later. The probes wire this service into the platform's self-healing; their precise design — why `/livez` must be shallow and `/readyz` may be deep — is the entire subject of the [health checks post](/blog/software-development/microservices/health-checks-readiness-liveness-and-self-healing), and you should treat that post as the spec for what goes behind these two URLs. The `resources` block is where the latency story from the intro lives, and we'll dissect it in its own section because it is the single highest-leverage and most-misunderstood part of the whole file.

### Service — a stable address and load balancing

Pods are mortal and their IPs change constantly. So how does the payment service find the order service? Not by Pod IP — that would be chasing a moving target. It finds it through a **Service**, which is a stable **virtual IP** (the *ClusterIP*) and DNS name that load-balances across whichever pods currently match its label selector. The Service is Kubernetes' built-in answer to [service discovery and load balancing](/blog/software-development/microservices/service-discovery-and-load-balancing) — the registry and the load balancer rolled into one platform primitive.

```yaml
apiVersion: v1
kind: Service
metadata:
  name: order-service
  namespace: shopfast
spec:
  type: ClusterIP            # default: reachable only inside the cluster
  selector:
    app: order-service       # selects the same pods the Deployment created
  ports:
    - port: 80               # the Service's stable port
      targetPort: 8080       # the container's port
```

Now any pod in the cluster can reach the order service at the DNS name `order-service.shopfast.svc.cluster.local` (or just `order-service` from within the same namespace), on port 80, and the platform load-balances each connection across the currently-ready pods. The Service IP never changes for the life of the Service, even as the pods behind it are created, killed, and rescheduled a thousand times. That stability is the whole point — it decouples *callers* from the churn of *instances*. We'll return to *how* that virtual IP actually works (kube-proxy, iptables/IPVS) in the networking section.

There are three Service types worth knowing. **ClusterIP** (the default) is internal-only — perfect for service-to-service traffic, which is most of your fleet. **NodePort** opens a high port on every node — crude, mostly for development or as a building block. **LoadBalancer** provisions a real cloud load balancer (an AWS NLB/ALB, a GCP LB) with an external IP — how a Service gets exposed to the public internet. In practice you rarely use a `LoadBalancer` per service; you use *one* and put an Ingress behind it.

### Ingress and the Gateway API — the front door

A `Service` of type `LoadBalancer` per public service would mean a separate (and separately-billed) cloud load balancer for each one. Instead you run **one** entry point and route by hostname and path. That's the **Ingress**: a single front door that terminates TLS and routes `shopfast.com/api/orders` to the order Service, `shopfast.com/api/payments` to the payment Service, and so on. This is the *north-south* edge (traffic in and out of the cluster) and it is exactly where your [API gateway and backend-for-frontend](/blog/software-development/microservices/the-api-gateway-and-backend-for-frontend) lives in a Kubernetes world.

```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: shopfast-edge
  namespace: shopfast
  annotations:
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
spec:
  ingressClassName: nginx
  tls:
    - hosts: ["shopfast.com"]
      secretName: shopfast-tls
  rules:
    - host: shopfast.com
      http:
        paths:
          - path: /api/orders
            pathType: Prefix
            backend:
              service:
                name: order-service
                port:
                  number: 80
          - path: /api/payments
            pathType: Prefix
            backend:
              service:
                name: payment-service
                port:
                  number: 80
```

An Ingress is only a *spec*; it does nothing until an **ingress controller** (NGINX, Traefik, HAProxy, or a cloud-native one) watches these objects and configures a real proxy accordingly. The newer **Gateway API** is the successor to Ingress — more expressive, role-oriented (a platform team owns the `Gateway`, app teams own `HTTPRoute`s), with first-class support for traffic splitting, header matching, and the canary routing you'll want in the [deployment strategies post](/blog/software-development/microservices/deployment-strategies-blue-green-canary-feature-flags). For a new cluster in 2026, prefer the Gateway API; the same `HTTPRoute` for the order service reads almost identically and gives you weighted backends for free.

### ConfigMap and Secret — configuration injected, not baked in

Your container image should be identical across dev, staging, and production; what *changes* is the configuration. Kubernetes gives you two objects for that: **ConfigMap** for non-sensitive config (feature flags, timeouts, the URL of a downstream service) and **Secret** for sensitive values (database passwords, API keys). Both can be injected as environment variables or mounted as files. The full discipline — rotation, encryption at rest, external secret stores, the fact that a base64 Secret is *not* encrypted — is the [configuration and secrets management post](/blog/software-development/microservices/configuration-and-secrets-management); here is the essential shape the Deployment above consumes.

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: order-service-config
  namespace: shopfast
data:
  PAYMENT_SERVICE_URL: "http://payment-service.shopfast.svc.cluster.local"
  REQUEST_TIMEOUT_MS: "800"
  MAX_DB_CONNECTIONS: "20"
  FEATURE_NEW_PRICING: "false"
---
apiVersion: v1
kind: Secret
metadata:
  name: order-service-secrets
  namespace: shopfast
type: Opaque
stringData:                 # stringData lets you write plaintext; k8s base64-encodes it
  db-password: "s3cr3t-rotate-me"
```

The Deployment above pulls the whole ConfigMap in via `envFrom` (every key becomes an env var) and pulls a single Secret key via `secretKeyRef`. The important production note, and a frequent surprise: **changing a ConfigMap does not restart the pods that consumed it as env vars.** Env vars are read once at container start. If you edit the ConfigMap, the running pods keep the old values until they restart. (Mounted-as-files ConfigMaps *do* update in place, eventually, but most apps read files once too.) The clean pattern is to roll the Deployment when config changes — annotate the pod template with a hash of the config so any change triggers a rollout. We'll see that in the optimization section.

### HorizontalPodAutoscaler — scale on load

The last essential object is the **HorizontalPodAutoscaler** (HPA). It is a controller that watches a metric — CPU utilization, memory, or a custom metric like requests-per-second — and *adjusts the Deployment's replica count* to keep that metric near a target. It is reconciliation applied to scaling: you declare "keep average CPU at 60%," and the HPA changes `replicas` up or down, after which the ordinary Deployment loop makes the new count real.

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: order-service
  namespace: shopfast
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: order-service
  minReplicas: 4
  maxReplicas: 40
  metrics:
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: 60     # target 60% of the CPU *request*
  behavior:
    scaleUp:
      stabilizationWindowSeconds: 0   # scale up immediately
      policies:
        - type: Percent
          value: 100                  # may double the pod count per step
          periodSeconds: 30
    scaleDown:
      stabilizationWindowSeconds: 300 # wait 5 min of calm before scaling down
```

Two subtleties that bite people. First, `averageUtilization: 60` is a percentage of the CPU *request*, not the limit and not the node's capacity. With a request of `500m`, 60% means the HPA targets 300m of average usage per pod and adds replicas when the fleet average climbs past that. If your requests are wrong, your autoscaling is wrong. Second, the asymmetric `behavior`: scale *up* fast (a flash sale won't wait) but scale *down* slowly (a 5-minute stabilization window prevents flapping — adding and removing pods every few seconds as traffic wobbles). We'll run real numbers through this in a worked example.

## The networking model: every pod gets an IP

Kubernetes networking has a deceptively simple foundational rule, and if you hold onto it the rest follows: **every pod gets its own unique, routable IP address, and every pod can reach every other pod's IP directly, with no NAT.** This is the "flat pod network." It means a pod is, network-wise, like a tiny VM — it has an IP, it listens on ports, and other pods dial it as if it were a host. A CNI plugin (Calico, Cilium, the cloud provider's own) makes this flat network real across all your nodes; you mostly don't think about it, but it's why the model is clean.

On top of that flat network sit the abstractions that make pods *findable* despite being mortal. The Service's ClusterIP is a **virtual** IP — there is no machine that owns it. Instead, `kube-proxy` (or, increasingly, eBPF in Cilium) programs every node's kernel with rules — iptables or IPVS — that intercept packets headed for the ClusterIP and rewrite the destination to one of the currently-ready pod IPs, chosen round-robin. When a pod becomes not-ready (fails its readiness probe) it's removed from that set; when it's deleted, gone. The "load balancer" is therefore distributed into the kernel of every node, with no central bottleneck. The figure traces a request from the public internet all the way down to a container port.

![A vertical stack tracing a request from the client through the Ingress controller to the Service ClusterIP then kube-proxy which picks a healthy pod IP and finally the container port](/imgs/blogs/kubernetes-for-microservices-the-essentials-5.webp)

The other half of discovery is **DNS**. Kubernetes runs an in-cluster DNS server (CoreDNS) that gives every Service a name. The order service's Service is reachable at `order-service.shopfast.svc.cluster.local`, and within the `shopfast` namespace you can shorten that to just `order-service`. So a service finds its dependencies by *name*, the name resolves to the stable ClusterIP, and the ClusterIP fans out to healthy pods. This is why your application code can simply `GET http://payment-service/charge` and never think about pod IPs, instance counts, or which node anything is on. The platform is doing client-agnostic service discovery for you — the same job you'd otherwise wire up with Consul or Eureka, here for free.

One honest caveat for the senior reader: ClusterIP load balancing is **L4 (connection-level)**, not L7 (request-level). For HTTP/1.1 that's usually fine — each request is often a fresh connection. But for **gRPC and HTTP/2**, which multiplex many requests over one long-lived connection, kube-proxy load-balances the *connection* once and then every request rides that same connection to that same pod. The result is lopsided load: one pod gets hammered, others idle. This is one of the concrete reasons teams reach for a [service mesh](/blog/software-development/microservices/service-mesh-istio-linkerd-when-you-need-one) (which does L7, per-request balancing) or client-side balancing. File it away; it's a real gotcha, not a theoretical one.

## Requests and limits: the #1 source of mystery latency

Here is the part of Kubernetes that causes more confusing, hard-to-diagnose production pain than anything else, and that almost everyone gets subtly wrong at first. It hinges on two fields you saw in the Deployment: `requests` and `limits`. They sound similar. They do completely different jobs, and conflating them is the root of the throttling and OOMKill stories.

A **request** is a *reservation*. When you say `requests: { cpu: 500m, memory: 512Mi }`, you are telling the scheduler "this pod needs half a core and 512MB *guaranteed* to even be placed." The scheduler uses requests, and only requests, to decide which node a pod fits on — it's a bin-packing problem, and requests are the box sizes. Requests also set the *floor*: the pod is guaranteed at least that much, and the HPA's "60% utilization" is measured against the CPU request.

A **limit** is a *ceiling*, and CPU and memory limits behave very differently when crossed, which is the crux:

- **CPU is compressible.** When a container tries to use more CPU than its limit, the kernel does not kill it — it **throttles** it. The CFS scheduler gives the container a quota per 100ms period (a `1000m` limit = 100ms of CPU per 100ms period across all cores), and once the quota is spent, the container is *paused* until the next period. Throttling doesn't crash anything; it just makes your code mysteriously slow in bursts. **This is the intro's latency ghost.** A service that needs a full core for 80ms but has a `500m` limit gets 50ms of CPU then sits frozen for 50ms — adding tens of milliseconds of latency that appears nowhere in your application metrics, only in `container_cpu_cfs_throttled_seconds_total`.

- **Memory is incompressible.** You cannot "throttle" memory — bytes are either allocated or they aren't. So when a container exceeds its memory limit, the kernel's OOM killer **terminates it immediately** with exit code 137. The pod shows `OOMKilled`, restarts, and if your real working set genuinely exceeds the limit, it OOMKills again — a `CrashLoopBackOff` spiral. **This is the intro's evil twin**, and we'll trace it event by event.

The relationship between requests and limits determines the pod's **Quality of Service (QoS) class**, which decides who gets killed first when a *node itself* runs out of memory. The figure lays out the three classes.

![A matrix comparing the Guaranteed Burstable and BestEffort QoS classes across whether requests equal limits and which pods are evicted first under node memory pressure](/imgs/blogs/kubernetes-for-microservices-the-essentials-7.webp)

- **Guaranteed**: requests *equal* limits for both CPU and memory. The pod gets exactly what it reserved, no more, and is the **last** to be evicted under node pressure. Best for latency-critical services where predictability beats burst.
- **Burstable**: requests are set but *lower* than limits. The pod is guaranteed its request but may burst up to its limit if the node has spare capacity. Evicted in the middle. The common, sensible default for most services.
- **BestEffort**: *no* requests and *no* limits. The pod gets whatever's left over and is the **first** to be killed when the node is squeezed. This is the "noisy neighbor" generator and almost always a mistake for anything you care about.

The senior heuristic, with numbers: **set memory request = memory limit** (so a memory spike never gets the pod OOMKilled by the kernel and never makes a node oversubscribed — memory is incompressible, you cannot safely overcommit it). **Set a CPU request that reflects steady-state usage, and either no CPU limit or a generous one** (so the service can burst into idle CPU rather than being throttled into latency). This is contrarian — many shops reflexively set CPU limits "to be safe" — but on a well-monitored cluster, removing CPU limits and relying on requests for fairness eliminates a whole class of mystery latency. We'll quantify that win in the optimization section.

#### Worked example: bin-packing 40 pods onto nodes

ShopFast's flash sale needs the order service at 40 replicas. Each pod requests `500m` CPU and `512Mi` memory. The cluster runs on nodes with 4 vCPU and 8GiB of RAM each. How many nodes does the order service alone need, and how much headroom is there?

First, subtract overhead. Each node loses about `500m` CPU and `1GiB` RAM to the kubelet, the OS, the CNI, and DaemonSets (log shippers, node exporters). So *allocatable* per node is roughly `3500m` CPU and `7GiB` RAM. Now bin-pack by the binding constraint:

- By CPU: `3500m / 500m = 7` pods per node.
- By memory: `7168Mi / 512Mi ≈ 14` pods per node.

CPU is the binding constraint at 7 pods/node, so 40 pods need `ceil(40 / 7) = 6` nodes (which hold 42 slots). Memory on those 6 nodes is `6 × 14 = 84` slots — so memory is half-empty while CPU is full. That imbalance is a signal: either the CPU request is too high (the service may not really need `500m` steady-state — measure it) or the node shape is wrong (CPU-heavy services want high-CPU node types). If profiling showed the order service actually averages `250m` under load, halving the request to `300m` (with headroom) gives `3500m / 300m ≈ 11` pods per node, dropping 40 pods from 6 nodes to `ceil(40/11) = 4` nodes. That's a **33% infrastructure saving on this one service**, achieved entirely by right-sizing one number, and it scales across every service in the fleet. This is why "set realistic requests" is not bookkeeping — it's the lever for your whole compute bill.

#### Worked example: an OOMKill at 256Mi, and the fix

The order service was set up early with `memory: { request: 256Mi, limit: 256Mi }` because that's what it used in a quiet dev environment. In production at steady state it sits at 230Mi — comfortable. Then the flash sale hits: request volume triples, the in-memory cache of hot products grows, and a burst of larger order payloads inflates per-request allocations. The real working set climbs to ~310Mi. The figure traces what happens next, second by second.

![A timeline of an OOMKill loop where a pod with a 256Mi limit grows its heap under a flash sale until the cgroup OOM killer terminates it with exit 137 and it enters CrashLoopBackOff](/imgs/blogs/kubernetes-for-microservices-the-essentials-9.webp)

The instant the container's memory crosses 256Mi, the kernel's cgroup OOM killer terminates the process — **exit code 137** (128 + signal 9, SIGKILL). `kubectl describe pod` shows `Last State: Terminated, Reason: OOMKilled`. The pod restarts, immediately starts climbing toward 310Mi again under the same load, and gets killed again. Because Kubernetes uses **exponential backoff** on restarts (10s, 20s, 40s, 80s, capped at 5 minutes), the pod enters `CrashLoopBackOff` and spends most of its time *not running*. Now you've lost capacity precisely when you need it most, and the surviving pods take more load, so *they* grow their working sets and start OOMKilling too — a fleet-wide cascade triggered by a single number being too small by ~60MiB.

The fix is to measure the true working set under peak load and set the limit above it with headroom. `kubectl top pod` or the `container_memory_working_set_bytes` metric shows the real peak. If peak is 310Mi, set the limit to `512Mi` (≈65% headroom over peak — generous, because memory is incompressible and the cost of OOMKill is total) and set the *request equal to the limit* so the pod is Guaranteed QoS and the scheduler reserves the full amount. The lesson: **memory limits must be derived from measured peak working set under realistic load, never from a quiet dev number.** A CPU misconfiguration makes you slow; a memory misconfiguration makes you dead.

## Namespaces and multi-tenancy

A **Namespace** is a logical partition of the cluster — a way to group objects (`shopfast`, `payments`, `staging`, `kube-system`) so names don't collide, RBAC permissions can be scoped, and resource budgets can be enforced. Two teams can both have an `order-service` Service as long as they're in different namespaces; DNS keeps them distinct (`order-service.shopfast` vs `order-service.checkout`).

For microservices, namespaces are how you do **soft multi-tenancy** and **cost control**. A `ResourceQuota` caps the total CPU/memory a namespace's pods may request, so one team can't accidentally consume the whole cluster. A `LimitRange` sets *default* requests/limits for any pod in the namespace that forgot to specify them — a cheap defense against the BestEffort noisy-neighbor problem. And `NetworkPolicy` objects restrict which namespaces and pods may talk to each other, the first step toward [zero-trust service-to-service security](/blog/software-development/microservices/service-to-service-security-mtls-and-zero-trust). A sensible default: one namespace per team or per bounded context, each with a quota, default limits, and network policies that deny cross-namespace traffic unless explicitly allowed.

```yaml
apiVersion: v1
kind: ResourceQuota
metadata:
  name: shopfast-quota
  namespace: shopfast
spec:
  hard:
    requests.cpu: "40"        # the namespace may request at most 40 cores total
    requests.memory: "80Gi"
    limits.cpu: "80"
    pods: "200"
```

A word of honesty for the senior: namespaces are *soft* isolation. They do not give you the hard, security-grade isolation between hostile tenants that separate clusters or VM-level boundaries (Kata Containers, gVisor) provide. Pods in different namespaces still share the same kernel and nodes. For trusted internal teams, namespaces are exactly right. For running genuinely untrusted code, you need a stronger boundary — that decision is a real fork, not a detail.

The practical reason to take namespaces seriously early is that they are the unit your *organization* maps onto the cluster, and that mapping is sticky. Conway's law shows up here directly: the namespace structure tends to mirror your team structure, because the natural ownership boundary for a `ResourceQuota`, an RBAC role, and an on-call rotation is a team. Get this wrong — one giant `default` namespace where every team's services pile together — and you lose the two things namespaces buy you. You lose **blast-radius control**, because a misconfigured pod from team A with no memory request can OOM a node hosting team B's critical service. And you lose **cost attribution**, because you can't tell whose pods are burning which cores when they're all in one bucket. The fix is cheap to apply on day one and expensive to retrofit once a hundred services share a namespace: one namespace per team or bounded context, each with a `ResourceQuota` so a team can't eat the cluster, a `LimitRange` so no pod is ever accidentally BestEffort, and default-deny `NetworkPolicy` so cross-team traffic is explicit. None of that is glamorous, and all of it is the difference between a cluster you can reason about and one where every incident is a forensic mystery about which tenant did what.

## Probes gate the rollout: tying health checks to deploys

You wrote three probes into the Deployment. Here is *why they're there from the platform's perspective*, which is different from the service's perspective covered in the [health checks post](/blog/software-development/microservices/health-checks-readiness-liveness-and-self-healing). The kubelet on each node polls each probe and turns the answers into actions:

- **Liveness** failing → **restart the container.** Use it only for "this process is wedged and a restart is the cure." It must be shallow — never check a dependency, or a dependency blip becomes a fleet-wide restart storm (the cardinal sin, dissected in depth in the health-checks post).
- **Readiness** failing → **remove the pod from the Service's endpoint set** (stop sending it traffic), but do *not* restart it. This is the gentle, reversible one, and it's the gate that makes rolling updates safe.
- **Startup** succeeding → **only then start running liveness/readiness.** It protects slow-booting apps (JVM warm-up, cache load) from being killed by an impatient liveness probe before they've finished starting.

The reason readiness *gates the rollout* is the linchpin of zero-downtime deploys. During a rolling update, Kubernetes will not consider a new pod "available" — and will not proceed to kill an old pod — until the new pod passes its **readiness** probe. So if your new version takes 12 seconds to warm its connection pool, readiness stays failing for 12 seconds, the new pod gets no traffic during that window, and the rollout *waits*. Without a readiness probe, Kubernetes considers a pod ready the instant its process starts, will route traffic to a not-yet-warm pod, and will happily tear down old pods before new ones can serve — dropping requests on every deploy. Readiness is the difference between a rolling update and a rolling outage.

## Rolling update mechanics: maxSurge and maxUnavailable

Let's trace exactly what happens when you `kubectl apply` a new image tag — `order-service:2.3.1` → `:2.4.0` — with the `maxSurge: 2, maxUnavailable: 0` strategy from the Deployment. The Deployment controller does *not* swap all six pods at once. It changes the desired state in bounded steps and lets reconciliation chase it, gated by readiness at every step. The figure walks the sequence.

![A timeline of a rolling update from six v1 pods through surging two v2 pods waiting for readiness draining two v1 pods and repeating in batches until six v2 pods are serving](/imgs/blogs/kubernetes-for-microservices-the-essentials-8.webp)

Step by step, with `maxSurge: 2` (up to 2 extra pods above the desired 6) and `maxUnavailable: 0` (never fewer than 6 ready):

1. Start: 6 v1 pods, all ready. Total serving capacity = 6.
2. The controller creates a new ReplicaSet for v2 and scales it to 2 (surge). Now there are 8 pods total: 6 ready v1 + 2 booting v2. Capacity stays at 6 because the v2 pods aren't ready yet.
3. The 2 v2 pods finish booting and **pass readiness**. Capacity is briefly 8.
4. *Now* the controller scales the v1 ReplicaSet down by 2. Those 2 v1 pods get a `SIGTERM`, flip their readiness to failing (draining), finish in-flight requests within the 30s grace period, and exit. Back to 6 ready (4 v1 + 2 v2).
5. Repeat: surge 2 more v2, wait for readiness, drain 2 v1.
6. End: 6 v2 pods ready, the v1 ReplicaSet scaled to 0 (kept around so `kubectl rollout undo` can instantly scale it back up). Total time ~90s, zero requests dropped, capacity never below 6.

The two knobs trade speed against safety and cost. `maxUnavailable: 0` guarantees full capacity throughout but requires headroom for the surge pods (you temporarily run 8 pods' worth of resources). `maxUnavailable: 1, maxSurge: 1` is cheaper (never more than 7 pods) but accepts dipping to 5 ready momentarily. For a capacity-critical service during peak, `maxUnavailable: 0` is right; for a roomy off-peak deploy, the cheaper setting is fine. And critically: **a rolling update is the floor, not the ceiling, of deployment safety.** It does not give you instant rollback of a *bad-but-healthy* version (the pods pass readiness; they just return wrong answers), nor canary analysis. That's the [deployment strategies post](/blog/software-development/microservices/deployment-strategies-blue-green-canary-feature-flags) — blue-green and canary build on these exact mechanics.

Here is the kubectl command set you actually use to drive and debug a rollout:

```bash
# apply the new desired state (the v2.4.0 image)
kubectl set image deployment/order-service order-service=registry.shopfast.internal/order-service:2.4.0 -n shopfast

# watch the rollout progress in real time
kubectl rollout status deployment/order-service -n shopfast

# see the new vs old ReplicaSets and pod counts
kubectl get rs -n shopfast -l app=order-service

# something's wrong — roll back to the previous version instantly
kubectl rollout undo deployment/order-service -n shopfast

# pause a rollout mid-flight (e.g. to inspect the first canary pods)
kubectl rollout pause deployment/order-service -n shopfast
kubectl rollout resume deployment/order-service -n shopfast

# why is a pod unhealthy? events, restart count, OOMKilled reason
kubectl describe pod -n shopfast <pod-name>

# what is it actually using right now? (catches throttling/OOM headroom)
kubectl top pods -n shopfast -l app=order-service

# tail logs from the previous (crashed) container instance
kubectl logs -n shopfast <pod-name> --previous

# get a shell inside a running pod to poke at it
kubectl exec -it -n shopfast <pod-name> -- /bin/sh
```

Those nine commands cover ~90% of day-to-day operating: deploy, watch, roll back, and debug. `kubectl describe pod` plus `kubectl logs --previous` is the OOMKill-debugging combo from the worked example — `describe` shows `Reason: OOMKilled` and the restart count; `--previous` shows what the killed process logged before it died.

### Reading a pod's life in `kubectl get pods`

A surprising amount of operating Kubernetes is just *reading the status column correctly*, because each state tells you exactly which part of the lifecycle is stuck. When you run `kubectl get pods`, the `STATUS` field is a compact diagnosis:

- **`Pending`** — the pod exists in etcd but isn't running yet. Almost always one of two things: the scheduler can't find a node with enough *requested* CPU/memory (your requests are too high or the cluster is full — `kubectl describe pod` shows `FailedScheduling: insufficient cpu`), or the image is still pulling. This is where over-large requests bite: a pod that requests more than any node can offer is *unschedulable forever*, silently sitting in `Pending`.
- **`ContainerCreating`** — scheduled, but the kubelet is still setting it up: pulling the image, mounting volumes, attaching the network. A pod stuck here usually means a slow/failed image pull (`ErrImagePull`, `ImagePullBackOff` — bad tag or missing registry credentials) or a volume that won't mount.
- **`Running`** but `READY 0/1` — the container is up but failing its **readiness** probe. It's alive, not serving traffic. This is the *normal* state during a slow boot and the *alarming* state if it persists — a dependency the readiness probe checks is down, or `/readyz` has a bug.
- **`CrashLoopBackOff`** — the container starts, exits, and Kubernetes is backing off before restarting it again. The cause is in `kubectl logs --previous`: an unhandled startup exception, a missing config, or — exit code 137 — an OOMKill. The "backoff" in the name is the exponential delay (10s → 20s → 40s → … → 5min cap) that keeps a crashing pod from hammering the node.
- **`OOMKilled`** (shown under `describe`'s Last State) — the memory-limit story; raise the limit from measured peak.
- **`Terminating`** that hangs — a pod stuck shutting down, usually because the app ignores `SIGTERM` and the kubelet is waiting out the full `terminationGracePeriodSeconds` before sending `SIGKILL`. Fix it in the app (handle the signal, drain, exit), not by deleting with `--force` in a loop.

Learning to map these five or six statuses to causes is most of what "debugging Kubernetes" means in practice for application teams; the deeper cluster-level forensics (etcd, scheduler decisions, CNI) is the platform team's beat, and the cross-service forensics — tracing a request across pods — is the [debugging distributed systems](/blog/software-development/microservices/debugging-distributed-systems-in-production) and [distributed tracing](/blog/software-development/microservices/distributed-tracing-and-observability-with-opentelemetry) posts.

## Optimization: making it production-grade with numbers

The single highest-leverage optimization in Kubernetes is **right-sizing requests and limits**, because requests drive your entire compute bill (via bin-packing) and limits drive your tail latency (via throttling) and your reliability (via OOMKill). Here is the production playbook, with the numbers that justify each move.

**Right-size CPU requests from measured usage.** Pull `container_cpu_usage_seconds_total` for a representative week and look at the p95 of per-pod CPU. If the order service requests `500m` but its p95 usage is `220m`, you're reserving more than double what it uses, halving your pods-per-node and doubling your node count. Drop the request to `300m` (p95 + headroom) and, per the bin-packing example, 40 pods go from 6 nodes to 4 — a 33% cut on this service. Across a fleet of forty services each over-requesting 2×, this is routinely the difference between a \$40,000/month and a \$25,000/month cluster bill. The metric to watch after the change: nodes' `Allocatable` CPU should sit around 70–80% requested (not 40%, which is waste; not 95%, which leaves no scheduling room).

**Remove or loosen CPU limits to kill throttling.** Check `container_cpu_cfs_throttled_seconds_total`. If a service is throttled more than ~5% of the time and its latency has unexplained bursts, its CPU limit is too tight. Removing the CPU limit (relying on requests for fair-share scheduling) on a service that was throttled 15% of periods commonly drops p99 latency by 20–40ms with no other change — that's the intro's bug, fixed. The trade-off is that an unbounded service *could* monopolize a node's spare CPU, which is why you do this on well-monitored clusters with sensible requests, not blindly.

**Set memory request = limit and derive both from measured peak.** Watch `container_memory_working_set_bytes` p99 under peak load, add ~50–65% headroom, and set request = limit at that value. This makes the pod Guaranteed QoS, prevents OOMKills, and prevents memory oversubscription (which is dangerous because memory can't be reclaimed by throttling). Target: zero OOMKills per week per service. One OOMKill is a signal to raise the limit, not a tolerable event.

**Bin-pack and spread deliberately.** Use **topology spread constraints** so your six order-service pods don't all land on one node (where a single node failure takes out your whole service) or all in one availability zone. A `maxSkew: 1` spread across zones means a zone outage costs you at most a third of your pods, which the surviving two-thirds and the HPA can absorb. The cost is slightly worse bin-packing (the scheduler can't pack as tightly), which is a price worth paying for surviving a zone failure.

```yaml
      topologySpreadConstraints:
        - maxSkew: 1
          topologyKey: topology.kubernetes.io/zone
          whenUnsatisfiable: ScheduleAnyway   # prefer spread, don't block scheduling
          labelSelector:
            matchLabels:
              app: order-service
```

**Tune the HPA to match traffic shape.** Default HPA reacts on a ~15s metric window and can lag a sharp spike. For a flash sale, set `scaleUp.stabilizationWindowSeconds: 0` and an aggressive `Percent: 100` policy (double pods per 30s step) so you reach capacity in a couple of minutes, and keep `scaleDown.stabilizationWindowSeconds: 300` to avoid thrashing as traffic wobbles down. Measure the win as **time-to-capacity** (how long from spike-detected to enough-pods-ready) and **rejected requests during scale-up** (which should trend to zero). For step-function traffic (a marketing email blast at a known time), schedule pods *ahead* with a CronJob that bumps `minReplicas` — reactive autoscaling always lags, and the cheapest fix is to not be reactive when you can predict.

#### Worked example: HPA scaling 4 → 40 during a flash sale

Steady-state, the order service runs at `minReplicas: 4`, each pod requesting `500m` CPU, averaging `180m` usage — that's 36% utilization, comfortably under the 60% target, so the HPA holds at 4. At 12:00 a marketing push triples traffic over ninety seconds. Per-pod CPU usage climbs toward `350m` — 70% of the request, over the 60% target. The HPA computes the desired replicas with its core formula:

`desiredReplicas = ceil(currentReplicas × currentMetric / targetMetric)` = `ceil(4 × 70 / 60)` = `ceil(4.67)` = **5**.

But traffic keeps climbing — by 12:02 each of the now-5 pods is again at ~70% because demand outran the small bump. The HPA recomputes each cycle and, with `Percent: 100` allowing it to double per step, ramps aggressively: 4 → 8 → 16 → 32 → and finally settles around **38–40** pods once average utilization falls back to ~58%. Total time from spike to stable capacity: roughly 2–3 minutes, bounded by how fast new pods boot and pass readiness (which is why a fast-booting, small image — the [containerizing post](/blog/software-development/microservices/containerizing-microservices-docker-best-practices) — directly improves your scale-up speed). When the sale ends at 12:30 and traffic drops, the 300-second scale-down stabilization window holds the fleet at ~40 for five quiet minutes before gently scaling back toward 4 — deliberately slow, so a brief dip doesn't trigger a costly scale-down-then-scale-back-up flap. The whole episode requires zero human action: you declared "keep CPU near 60%, between 4 and 40 pods," and reconciliation did the rest.

## Stress-testing the design

A design is only as good as its behavior at the edges. Let's poke ShopFast's setup the way production will.

**A node dies — what happens to its pods?** Say a node holding 7 order-service pods hardware-fails. The kubelet stops reporting; after `node-monitor-grace-period` (~40s by default) the node is marked `NotReady`, and after a further eviction timeout (~5 min by default, often tuned shorter) its pods are marked for deletion. The Deployment controller sees actual replicas drop below desired and **immediately creates replacement pods** on healthy nodes — it does not wait for the dead node's pods to be cleanly removed. So within seconds-to-a-minute of the node being declared dead, 7 new pods are scheduling elsewhere. The Service's endpoint set drops the dead pods (they fail readiness / are unreachable) so traffic stops flowing to them. The user-visible impact: a brief blip of in-flight requests to that node failing (which your client-side [retries and timeouts](/blog/software-development/microservices/resilience-patterns-timeouts-retries-circuit-breakers-bulkheads) absorb), then full recovery. The lesson: the *application* must tolerate a sudden loss of a chunk of instances — that's resilience patterns in your code, not something Kubernetes does for you. And the eviction *timeout* matters: the default 5-minute pod-eviction delay means traffic could route toward dead pods for minutes if readiness weren't catching them. Spread (topology constraints) is what keeps "a node died" from meaning "my whole service died."

**Memory limit too low → OOMKill loop.** Covered in the worked example: the pod crosses its limit, gets killed with exit 137, restarts, climbs again, and `CrashLoopBackOff` removes it from service precisely when load is highest, cascading to siblings. The defense is measured limits with headroom and an alert on *any* OOMKill, treated as a sev rather than noise.

**No requests set → noisy neighbor.** A pod with no CPU/memory requests is **BestEffort** QoS. The scheduler thinks it needs nothing, so it packs it onto an already-busy node. Then under load it competes for CPU with its neighbors (no reserved floor) and, when the node hits memory pressure, it's the **first** thing the kubelet evicts. Worse, a *missing memory request* on a greedy pod lets it balloon and trigger node-level OOM that can take down *other, well-behaved* pods on the same node. This is the classic "one team's misconfigured batch job paged the whole cluster" story. The fix is structural: a `LimitRange` in every namespace that sets default requests/limits so no pod is ever accidentally BestEffort.

**A downstream is slow — does Kubernetes save you?** No, and this is the most important honesty in the post. Kubernetes restarts crashed pods, scales on load, and load-balances connections. It does **not** add timeouts to your outbound calls, retry failed requests, open circuit breakers, or shed load. If the payment service goes slow, your order-service pods will block threads waiting on it, exhaust their pools, fail readiness, get pulled from rotation, and — if you put the dependency check in the *liveness* probe — get *restarted* into a storm. Kubernetes gives you the platform; the in-process resilience ([timeouts, retries, circuit breakers, bulkheads](/blog/software-development/microservices/resilience-patterns-timeouts-retries-circuit-breakers-bulkheads)) is your code's job, or the [service mesh](/blog/software-development/microservices/service-mesh-istio-linkerd-when-you-need-one)'s. Knowing where Kubernetes' responsibility ends is exactly what separates a senior from someone who thinks "we're on k8s, so we're resilient."

## Trade-offs: where to run a service fleet

Kubernetes is one option, not the only one, and a senior names the cost before recommending it. The honest question is "where should this fleet run?" — and the answer depends on your team size, traffic shape, and how much operational complexity you can carry. The figure compares the four realistic homes for a service fleet.

![A matrix comparing raw VMs PaaS ECS or Nomad and Kubernetes across operational cost flexibility auto-scaling and learning curve](/imgs/blogs/kubernetes-for-microservices-the-essentials-6.webp)

| Option | You gain | You pay | When it wins |
|---|---|---|---|
| **Raw VMs + scripts** | Total control, nothing hidden | All the operational tax by hand; reinventing scheduling, restarts, scaling | A handful of long-lived services, a strong ops team, regulatory constraints on managed platforms |
| **PaaS (Heroku, Render, Fly, App Runner)** | Near-zero ops; deploy a container and forget infra | Less control, higher per-unit cost at scale, vendor lock-in, scaling ceilings | Small teams, early-stage products, services where engineer time dwarfs infra cost |
| **Managed containers (ECS Fargate, Nomad)** | Container orchestration without running a control plane | Less of an ecosystem than k8s; some cloud lock-in (ECS) | Mid-size fleets that want orchestration but not the full Kubernetes operational surface |
| **Kubernetes (EKS/GKE/AKS or self-run)** | Maximum automation, flexibility, portability, ecosystem | Steepest learning curve; real platform-team cost; many sharp edges | Large or fast-growing fleets, multiple teams, need for portability and a rich ecosystem |

The decisive framing: **Kubernetes pays off when the size of your fleet makes the operational tax dominate, and you have (or can build) a platform team to own the cluster.** A managed control plane (EKS/GKE/AKS) removes the worst of running Kubernetes itself, but you still own the workloads, the YAML, the upgrades, the networking, and the on-call. For a team of five running three services, that overhead is pure cost with little benefit — a PaaS will be faster and cheaper in *total* (including engineer time) for years. For a company of two hundred engineers running a hundred services, the per-service automation and the shared platform make Kubernetes clearly worth it. The mistake juniors and over-eager architects make is choosing Kubernetes for the résumé or the future, paying its full cost on day one for a benefit that only materializes at a scale they're nowhere near.

## Case studies

**A mid-size company's migration: the cost showed up in the platform team, the benefit in deploy velocity.** A recurring, well-documented pattern across engineering blogs (and one I've seen first-hand) is a company moving from VMs-plus-scripts to managed Kubernetes and finding the headline benefit is *deployment velocity and density*, while the headline cost is a *standing platform team*. The win is concrete: deploys that took a coordinated 90-minute change window become self-service rolling updates any team runs in minutes, and bin-packing many services per node cuts the VM count meaningfully (often 30–50% fewer instances than one-service-per-VM). The cost is equally concrete: Kubernetes is not free to operate — upgrades, networking, RBAC, observability, and the cluster's own failure modes need dedicated owners. The lesson seniors take: budget for the platform team *as part of the decision*, not as a surprise six months in. Kubernetes shifts toil from every product team into one platform team; that's a good trade at scale and a bad one below it.

**The requests-and-limits throttling gotcha at scale — the latency that isn't in your code.** The intro's story is not hypothetical; CPU-throttling-as-mystery-latency is one of the most common large-scale Kubernetes incidents, written up repeatedly by teams running big fleets. The pattern is always the same: a service with a CPU limit gets throttled under load, p99 latency spikes in bursts, every application-level metric looks fine, and engineers burn hours before someone checks `container_cpu_cfs_throttled_seconds_total`. Several large engineering orgs have publicly described removing CPU limits across their fleets (keeping requests for fairness) and seeing tail latency drop with no downside on well-monitored clusters. There was even a notorious Linux kernel CFS bug (fixed in 2019) that *over*-throttled containers that weren't actually over their quota — entire companies saw latency improvements from a kernel patch. The lesson: limits are a sharp tool, CPU and memory limits behave oppositely, and "set a limit to be safe" is folk wisdom that frequently makes things worse.

**Monzo runs ~1,500 microservices on Kubernetes — but they built the platform muscle to match.** The UK bank Monzo is the canonical example of microservices-at-extreme-granularity on Kubernetes, with well over a thousand services on a single platform. What matters for *this* post is the second half of that story: it works *because* Monzo invested heavily in platform tooling, code generation, strict service templates, and a dedicated platform team — the per-service operational tax is automated almost to zero by tooling on top of Kubernetes. The lesson is the inverse of the migration case: Kubernetes makes a huge fleet *possible*, but the platform investment is what makes it *sane*. You don't get Monzo's velocity by installing Kubernetes; you get it by building the layer of automation and convention on top that turns "1,500 services" from a nightmare into a routine.

**A team that did NOT need Kubernetes — and was right.** The most useful case study is a negative one. A small startup — say six engineers, three services, modest traffic — that deliberately ran on a PaaS (Heroku, Render, or Fly) instead of Kubernetes. They shipped features instead of running a control plane, had zero cluster-upgrade incidents (because there was no cluster), and paid a higher per-unit infrastructure cost that was trivial next to the engineer-months they didn't spend on platform work. When they eventually outgrew the PaaS — multi-region needs, cost at scale, the desire for more control — *then* they migrated to managed Kubernetes, with real workloads and real numbers to justify it. The lesson seniors take: "we'll need Kubernetes eventually" is not a reason to adopt it now. Adopt the simplest thing that runs your fleet today, and let real scale, not anticipated scale, pull you up the complexity ladder. The migration later is cheaper than the operational tax of running Kubernetes for years before you needed it.

## When to reach for Kubernetes (and when not to)

Reach for Kubernetes when **most of these are true**: you run more than a handful of services; multiple teams deploy independently and often; you need automated scaling, self-healing, and zero-downtime rolling deploys as a default rather than a project; you value portability across clouds or want a rich ecosystem (operators, Helm, the mesh, GitOps); and you have — or can fund — a platform team to own the cluster. At that point Kubernetes' automation of the operational tax clearly outweighs its complexity, and the alternative (scripts and heroics) is the bigger risk.

Do **not** reach for Kubernetes when: you have a small team and a few services that a PaaS would run with a fraction of the effort; your traffic is steady and modest, so the autoscaling and density wins are small; you have no one to own the cluster, its upgrades, and its on-call; or you're choosing it for the résumé, the future, or because "everyone uses it." For a startup, a PaaS or managed-container service (Fargate, Cloud Run, Render) will almost always ship faster and cost less *in total* — and you can migrate to Kubernetes later, with real numbers, when scale actually pulls you there. The honest senior position: Kubernetes is a phenomenal tool for the problem it solves, the operational tax of a large service fleet — and an expensive, complexity-adding mistake for teams that don't have that problem yet. Match the tool to the problem you actually have today, not the one you expect to have someday.

There's a second-order trap worth naming explicitly, because it catches good engineers. Kubernetes is *so* capable that, once you have it, every problem starts to look like a Kubernetes problem, and the cluster accretes complexity — custom operators, a service mesh, three layers of CRDs, a GitOps toolchain — each addition individually justified and the sum a system only two people understand. The discipline is to keep asking, of every new piece, the same question you ask of Kubernetes itself: what concrete, present problem does this solve, what does it cost to operate, and who owns it at 3am? A cluster running six objects per service that the whole team understands is worth more than a gleaming platform that's a black box to everyone but the one engineer who built it and is now interviewing elsewhere. Power you can't operate confidently is a liability, not an asset.

And the layer *above* Kubernetes: when plain Service-level (L4) load balancing, in-code resilience, and per-pod observability stop being enough — when you need uniform mTLS between services, L7 (per-request) load balancing for gRPC, traffic-shifting for canaries, and a single pane of golden-signal metrics without instrumenting every service — that's when a **service mesh** earns its (considerable) cost. That's the [next post](/blog/software-development/microservices/service-mesh-istio-linkerd-when-you-need-one), and the rule there mirrors the rule here: a mesh is powerful and heavy, so adopt it for a real problem, not a hypothetical one.

## Key takeaways

1. **Kubernetes automates the operational tax** — scheduling, restarts, scaling, service discovery, rolling deploys — that every service in a fleet must pay. Its value grows with the number of services, which is exactly why it fits microservices and overshoots a small app.
2. **Reconciliation is the one idea that matters.** You declare desired state; controllers continuously drive actual toward desired. Self-healing, rolling updates, and autoscaling are all just this one loop. Work with the loop (change the declared state, let it reconcile); never patch a live object by hand.
3. **Learn six objects, not six hundred.** Pod (the unit), Deployment (replicas + rollouts), Service (stable virtual IP + L4 balancing), Ingress/Gateway (north-south entry), ConfigMap/Secret (config), and HPA (scale on load). They compose into one stack per service.
4. **Requests are reservations; limits are ceilings — and CPU and memory limits behave oppositely.** Over the CPU limit, you're *throttled* (mystery latency). Over the memory limit, you're *OOMKilled* (exit 137, CrashLoopBackOff). Set memory request = limit from measured peak; keep CPU requests honest and CPU limits loose-or-absent.
5. **Requests drive your bill; limits drive your latency and reliability.** Right-sizing one CPU request can cut a service's node count by a third; one too-low memory limit can cascade a whole service into a crash loop. These numbers are the highest-leverage tuning you'll do.
6. **Readiness gates the rollout.** Without a readiness probe, every deploy can drop requests; with one, a rolling update with `maxUnavailable: 0` is genuinely zero-downtime. Probes are the contract between your service and the platform's self-healing.
7. **Kubernetes is not your resilience layer.** It restarts, scales, and balances — it does not add timeouts, retries, circuit breakers, or load shedding. Those live in your code (or the mesh). Knowing where the platform's job ends is the senior's edge.
8. **Choose the simplest home that runs today's fleet.** A PaaS or managed-container service beats Kubernetes in total cost for small teams; Kubernetes wins at fleet scale *with* a platform team. "We'll need it eventually" is not a reason to pay its cost now.

## Further reading

- *Kubernetes Up & Running* by Burns, Beda, Hightower, and Evenson — the most practical on-ramp to the objects and the operational model.
- *Programming Kubernetes* by Hausenblas and Schimanski — for the reconciliation/controller model in depth, once the loop intrigues you.
- The official [Kubernetes documentation](https://kubernetes.io/docs/) — especially the Concepts sections on Pods, Deployments, Services, and Resource Management; the canonical, accurate reference.
- The [Kubernetes resource management and QoS docs](https://kubernetes.io/docs/concepts/workloads/pods/pod-qos/) — read this twice; it's the foundation of the requests/limits story.
- *Building Microservices* by Sam Newman (2nd ed.) — the deployment and "should you even" chapters frame Kubernetes in the wider microservices context honestly.
- The [Gateway API documentation](https://gateway-api.sigs.k8s.io/) — the successor to Ingress; prefer it for new clusters.
- This series' siblings: [containerizing microservices](/blog/software-development/microservices/containerizing-microservices-docker-best-practices) (the prerequisite), [health checks](/blog/software-development/microservices/health-checks-readiness-liveness-and-self-healing) (what goes behind the probes), [service discovery and load balancing](/blog/software-development/microservices/service-discovery-and-load-balancing) (what Services automate), [the API gateway and BFF](/blog/software-development/microservices/the-api-gateway-and-backend-for-frontend) (what Ingress hosts), and the next steps: [service mesh](/blog/software-development/microservices/service-mesh-istio-linkerd-when-you-need-one), [deployment strategies](/blog/software-development/microservices/deployment-strategies-blue-green-canary-feature-flags), and [configuration and secrets management](/blog/software-development/microservices/configuration-and-secrets-management).
