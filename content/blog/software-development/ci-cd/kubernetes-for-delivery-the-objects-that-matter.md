---
title: "Kubernetes for Delivery: The Objects That Matter"
date: "2026-06-22"
publishDate: "2026-06-22"
description: "You do not need to know all of Kubernetes to ship to it. You need the handful of objects that ARE your deployment and one idea — declarative desired state driven by a reconcile loop. This post is the delivery view of k8s: the Deployment, Service, Ingress, ConfigMap, and Secret you kubectl apply to put a service in production, and why that declarative model is the thing that makes self-healing and GitOps work."
tags:
  [
    "ci-cd",
    "devops",
    "kubernetes",
    "deployment",
    "declarative",
    "reconcile-loop",
    "gitops",
    "manifest",
    "self-healing",
    "everything-as-code",
  ]
category: "software-development"
subcategory: "CI/CD & Delivery"
author: "Hiep Tran"
featured: true
readTime: 56
image: "/imgs/blogs/kubernetes-for-delivery-the-objects-that-matter-1.png"
---

The first time I shipped a service to Kubernetes, I tried to read the whole thing first. I bought the book. I learned what the scheduler does, how the CNI plugin wires pod networking, the difference between the kubelet and the kube-proxy, what etcd is and why it uses Raft. Three weeks later I knew a great deal of trivia and I still had not deployed a single thing. The cluster was sitting there, healthy and empty, waiting for me to feel ready.

Here is what I wish someone had told me on day one: you do not deploy the scheduler. You do not write the CNI. You do not touch etcd. Those are the *platform* — somebody (a cloud provider, a platform team, you-on-a-different-day) stood them up once, and they run themselves. The thing you actually do, every single deploy, is write a small file that says "I want three copies of this exact image, reachable at this URL, with this config," and hand it to the cluster. That file — a Deployment plus a Service plus an Ingress, maybe a ConfigMap — *is* your deployment. Everything else is the machine that reads it.

That is the entire reframe of this post. Kubernetes looks enormous because the reference documentation describes a distributed operating system. But the *delivery surface* — the part you `kubectl apply` to ship a service — is roughly five objects and one idea. The idea is the load-bearing one, so I will state it up front and spend the rest of the post earning it: **you do not run deploy scripts; you declare a desired state, and a control loop continuously drives the live cluster to match it.** A pod dies at 3am? The loop notices the gap and makes a new one. A node falls over? The loop reschedules its pods elsewhere. You were asleep. Nothing paged you. That self-healing is not a feature you turn on — it is the direct consequence of the declarative model, and it is also exactly why GitOps (Git as the source of truth, covered later in this series) works at all.

![A layered diagram showing a manifest in Git flowing through kubectl apply to the API server, then a reconcile loop observing and diffing actual state until the cluster converges on the desired state](/imgs/blogs/kubernetes-for-delivery-the-objects-that-matter-1.png)

By the end of this you will be able to read and write a complete Deployment, Service, and Ingress manifest, wire them together with labels, ship them with `kubectl apply` and `kubectl rollout status`, watch the cluster heal itself when you break it on purpose, and — crucially — know which parts of Kubernetes are *yours* to version in Git and which parts are the platform's problem. This is the delivery view. For how the scheduler places pods, how the CNI moves packets, and how to architect a fleet of services, I will point you at the right neighbors as we go rather than re-derive them here. This post sits on the `commit → build → test → package → deploy → operate` spine of the whole series at the **deploy** step: the artifact is already built, scanned, and pinned by digest; now we put it on the cluster.

## 1. The one idea: declarative desired state and the reconcile loop

Most tools you have used are *imperative*. You run a command and it does a thing, once. `cp this that`. `systemctl restart nginx`. `./deploy.sh`. The command executes, returns, and forgets. If the thing it did gets undone five minutes later — the process crashes, the file gets deleted — the command does not come back and fix it. It already ran. It is over.

Kubernetes is *declarative*. You do not tell it what to *do*; you tell it what you *want* to be *true*. "I want three replicas of `web:sha256:abc…` running, reachable as `web` inside the cluster, exposed at `app.example.com`." You write that down as a manifest — a YAML file describing objects — and you submit it. The cluster stores your statement of intent (in etcd, via the API server) as the **desired state**. Then a set of controllers — small control loops, one per kind of object — wake up over and over, look at the **actual state** of the world, compare it to the desired state, and take whatever action closes the gap. This compare-and-correct cycle is the **reconcile loop**, and it never stops.

The figure above is the loop drawn out: your manifest in Git, `kubectl apply` submitting it to the API server which stores the desired state, and the reconcile loop continuously observing actual state, diffing it against desired, and acting until the two converge. The arrow that matters is the one that loops back forever. You declared once; the loop enforces continuously.

Put plainly: an imperative deploy is a *verb* you execute. A declarative deploy is a *noun* you assert into existence and the system keeps true. The difference sounds academic until something breaks. Suppose you ran `./deploy.sh` to start three containers, and at 3am one of them OOM-kills itself. With the imperative model, you now have two containers and a script that finished hours ago. Nothing brings the third one back unless a human or a separate watchdog notices. With the declarative model, the controller's next reconcile pass sees "desired 3, actual 2," and it schedules a replacement. No human. No script re-run. The gap simply closes because closing gaps is the loop's only job.

This is the foundation of cloud-native delivery, and it inverts how you operate. You stop thinking "what commands do I run to get from the current state to the new state?" and start thinking "what state do I *want*, and how do I edit the declaration?" To deploy a new version, you do not run a sequence of steps — you change one line in the manifest (the image digest) and re-apply. The loop figures out the steps. To scale up, you change `replicas: 3` to `replicas: 10` and apply. The loop creates seven pods. To roll back, you re-apply the previous manifest. The loop converges back.

### Why this is the thing that makes GitOps work

Hold onto the declarative model for a second longer, because it pays off twice. If your entire deployment is a set of declarative manifests, then the *truth* about what should be running is just a pile of YAML files. And YAML files belong in Git. So you can put the desired state in a Git repository, and now Git is the single source of truth: the live cluster *should* match `main`, and a tool (Argo CD or Flux) continuously pulls from Git and reconciles the cluster toward it — the same compare-and-correct loop, just with Git as the desired-state store instead of a manual `kubectl apply`. That is **GitOps**, and it only works *because* Kubernetes is declarative. You cannot do GitOps with imperative deploy scripts — there is no "desired state" to diff against. I will not build out GitOps here; that is a later post in this series (`gitops-git-as-the-source-of-truth`). But every time you write a manifest in this post, notice that it is a plain text file you could commit, review in a pull request, and let a controller apply. That property is not an accident — it is the whole point.

The honest caveat: the loop is *eventually* consistent, not instant. There is latency between "desired changed" and "actual matches" — the controller has to notice, the scheduler has to place pods, images have to pull, probes have to pass. For a routine reconcile that is seconds; for a rollout of a fat image onto cold nodes it can be minutes. The loop also cannot fix what it cannot express: if your container crashes on startup because of a bad config, the loop will dutifully restart it forever (CrashLoopBackOff) — it is faithfully driving toward a desired state that is itself broken. Declarative is powerful, not magic. It enforces *your* intent, including your mistakes.

### Why the declarative model moves the DORA numbers

The series measures everything against the four DORA metrics — deploy frequency, lead time for changes, change-failure rate, and time-to-restore. It is worth being precise about *why* moving the deploy onto a declarative reconcile loop pushes those numbers in the right direction, because "it's modern" is not an argument and an engineer should be able to show the mechanism.

Take **time-to-restore (MTTR)** for the most common production failure: a single replica dying. In the imperative world MTTR for that event decomposes into roughly four terms — time-to-detect, time-to-acknowledge (a human, often asleep, sees the page), time-to-diagnose, and time-to-act. Call it $T_{detect} + T_{ack} + T_{diag} + T_{act}$. Realistically that is something like 2 minutes to alert plus 8 minutes for a paged engineer to wake and acknowledge plus 5 minutes to diagnose plus 2 minutes to restart — about 17 minutes, and at 3am the acknowledge term alone can be 20+. The declarative model deletes the entire human chain for this class of failure: the loop's detect-and-act cycle is on the order of seconds, so MTTR for "a pod died" drops from ~17 minutes to ~5 *seconds*. That is not a 2x improvement; it is a category change, because you removed the human from the loop for the failures the loop can handle. You still keep the human for the failures it cannot (a bad deploy, lost capacity) — which is exactly the right division of labor.

Now **change-failure rate.** A meaningful slice of deploy failures in hand-run procedures is not bad *code* — it is bad *execution* of the deploy: the wrong host order, a missed flag, a half-finished rollout, a config edited on one host but not the others. Call that the "mechanics" component of your change-fail rate, separate from the "genuine bug" component. The declarative model drives the mechanics component toward zero, because the same deterministic controller performs the identical rollout every time from the same declared spec — there is no host order to get wrong and no flag to forget. If your change-fail rate was 18% and two-thirds of those failures were mechanics, eliminating mechanics takes you to roughly 6% with no change to your code quality at all. The residual is genuine bugs, which is a *testing* problem (see the sibling `the-test-stage-fast-feedback-vs-confidence`), not a *deploy* problem — and now you can see them clearly instead of having them masked by deploy noise.

**Lead time and deploy frequency** move because the act of deploying collapses from a multi-step ceremony to a one-line diff and one `apply` that a pipeline can run unattended. When a deploy is a 14-step runbook requiring two people and a maintenance window, you batch changes and ship monthly — large batches, scary deploys, high failure rate, the whole vicious cycle the DORA research documents. When a deploy is "change the digest, `kubectl apply`, watch `rollout status`," a pipeline does it on every merge, and you ship many small changes a day. Small batches are the lever: a 30-line diff is easy to review, test, and revert; a 3,000-line monthly batch is not. The declarative deploy is what makes small batches *cheap enough* to be the default, and the DORA data is unambiguous that small-batch, high-frequency teams have *lower* change-fail rates and *shorter* restore times, not higher — speed and stability move together, they are not a trade-off. The declarative model is one of the structural reasons that is true.

I will not pretend Kubernetes alone delivers these numbers — it is the deploy substrate, and you still need a real pipeline, real tests, and progressive delivery on top (all elsewhere in this series). But the *deploy* link in the chain, re-expressed as declared-state-plus-reconcile, is a measurable improvement on every DORA axis, and the reasons are mechanical, not magical.

## 2. Pods: the unit that runs (but rarely the unit you write)

The smallest thing Kubernetes runs is a **Pod**. A pod is one or more containers that share a network namespace (same IP, same localhost, same port space) and can share storage volumes. Most pods are a single container; the multi-container pattern (a "sidecar" — a log shipper, a service-mesh proxy, a config reloader next to your app) exists, but you can ignore it until you need it. The mental shorthand that gets you 90% of the way: a pod is "a running instance of your container, with an IP."

Two properties of pods drive every delivery decision downstream, so internalize them:

**Pods are ephemeral.** A pod is cattle, not a pet. It can be killed and recreated at any time — by a rolling update, by a node draining for maintenance, by the scheduler rebalancing, by an OOM kill. When a pod dies and a new one takes its place, the new pod gets a *new IP*. Nothing about the old pod survives unless you explicitly persisted it to a volume. This is why you never, ever hardcode a pod's IP anywhere — it will be wrong within hours. (The Service object, section 4, exists precisely to give you a stable address in front of these moving targets.)

**You rarely create pods directly.** You *can* write a bare Pod manifest, but you almost never should, because a bare pod has no one watching it. If it dies, it stays dead — you are back in the imperative-script world. Instead, you declare a *controller* (a Deployment) that says "keep N pods like this template alive," and let the controller create and recreate the pods for you. The pod is the unit of *running*; the Deployment is the unit of *shipping*. You write Deployments. The Deployment writes pods.

Resource requests and limits live on the pod's containers, and they are part of the deployable spec — not an afterthought. A `requests` value tells the scheduler how much CPU and memory to reserve when placing the pod (it is the basis for bin-packing nodes); a `limits` value is the ceiling the kernel enforces. Those limits become Linux cgroup constraints on the container process — the same cgroup mechanism that bounds any container, which the earlier container post in this series covers at the kernel level. The delivery-relevant point: if you omit requests, the scheduler is flying blind and may overpack a node into instability; if you omit a memory limit, one leaky pod can starve its neighbors. Declaring resources is declaring your deployable's contract with the platform.

```yaml
# A Pod spec almost always lives INSIDE a Deployment's template,
# not as a standalone object. Shown here just to see the shape.
apiVersion: v1
kind: Pod
metadata:
  name: web-demo
  labels:
    app: web
spec:
  containers:
    - name: web
      image: ghcr.io/acme/web@sha256:9f2b...c41a   # pinned by digest, not :latest
      ports:
        - containerPort: 8080
      resources:
        requests:
          cpu: "100m"        # reserve 0.1 CPU for scheduling
          memory: "128Mi"
        limits:
          cpu: "500m"
          memory: "256Mi"    # cgroup hard ceiling; exceed -> OOM kill
      readinessProbe:
        httpGet:
          path: /healthz
          port: 8080
        initialDelaySeconds: 3
        periodSeconds: 5
```

Notice the image is pinned by `@sha256:…` digest, not a mutable tag like `:latest`. This is the **build-once, promote-everywhere** discipline of the whole series showing up at the cluster: the exact bytes you built, scanned, and tested are the exact bytes that run, with no possibility of a tag silently pointing somewhere new between staging and prod. (See the sibling post `build-once-promote-everywhere-artifacts-and-versioning` for why this matters and how to wire it.) And notice the `readinessProbe` — the cluster will not send traffic to this pod until `/healthz` returns 200. That probe is what gates a rolling update, which becomes the spine of section 6.

## 3. Deployment: the object you actually ship

The **Deployment** is the object you write and the object you change to deploy. It declares three things: how many replicas you want, the pod template (the spec from section 2, embedded), and the update strategy (how to roll from the old version to the new one). When you submit a Deployment, a chain of controllers springs into action: the Deployment controller creates a **ReplicaSet** (an object whose entire job is "keep exactly N pods matching this template alive"), and the ReplicaSet controller creates the pods. You declared a Deployment; you got a ReplicaSet; you got pods. Three levels, one `apply`.

![A branching graph showing a Deployment owning a ReplicaSet that owns two Pods labeled app web, with a Service selecting those Pods and an Ingress routing external users into the Service](/imgs/blogs/kubernetes-for-delivery-the-objects-that-matter-2.png)

The figure shows the ownership chain and how the other objects attach by label. The Deployment owns the ReplicaSet; the ReplicaSet owns the pods; the Service and Ingress *find* the pods by matching labels (more on that wiring in section 4). The key behavior for delivery is the part that does not show in the diagram: **when you bump the image in the Deployment and re-apply, the Deployment controller creates a NEW ReplicaSet for the new template and gradually scales it up while scaling the old one down.** That gradual swap is a rolling update — your zero-downtime deploy mechanism, and the subject of the dedicated sibling post `rolling-updates-and-zero-downtime-deploys`. Here, the relevant facts are: the rollout is driven by the reconcile loop (you do not script the steps), and it is gated by the readiness probe (the loop will not retire an old pod until a new one reports ready).

Here is a real Deployment for a web service. Read it as a declaration of intent, not a script:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: web
  namespace: shop
  labels:
    app: web
spec:
  replicas: 3
  selector:
    matchLabels:
      app: web            # this Deployment manages pods with label app=web
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxUnavailable: 0    # never drop below desired capacity during a roll
      maxSurge: 1          # add at most 1 extra pod at a time
  template:
    metadata:
      labels:
        app: web           # the label that wires Service/Ingress to these pods
    spec:
      containers:
        - name: web
          image: ghcr.io/acme/web@sha256:9f2b...c41a
          ports:
            - containerPort: 8080
          envFrom:
            - configMapRef:
                name: web-config
            - secretRef:
                name: web-secrets
          resources:
            requests: { cpu: "100m", memory: "128Mi" }
            limits:   { cpu: "500m", memory: "256Mi" }
          readinessProbe:
            httpGet: { path: /healthz, port: 8080 }
            initialDelaySeconds: 3
            periodSeconds: 5
          livenessProbe:
            httpGet: { path: /livez, port: 8080 }
            initialDelaySeconds: 10
            periodSeconds: 10
```

A few things to read closely. The `selector.matchLabels` and the `template.metadata.labels` must agree — the Deployment manages exactly the pods carrying `app: web`. The `strategy` block with `maxUnavailable: 0` and `maxSurge: 1` is the rollout safety knob: it tells the loop to bring up a new pod *before* taking an old one down, so you never dip below three serving replicas. The `envFrom` block pulls configuration and secrets from a ConfigMap and a Secret (section 7) rather than baking them into the image — same image, different config per environment, which is what makes one artifact promotable. And the two probes do different jobs: `readinessProbe` controls whether the pod *receives traffic* (gates the rollout and the Service endpoint list); `livenessProbe` controls whether the kubelet *restarts the container* (it detects a wedged process). Confusing these two is one of the most common and most painful k8s delivery bugs — a liveness probe that is really a readiness check will restart a healthy-but-busy pod into a crash loop.

#### Worked example: what a deploy actually is

A team I worked with had a 14-step runbook for deploying their service: SSH to each of three hosts, `docker pull`, `docker stop`, `docker rm`, `docker run` with a 9-flag command, check the logs, repeat. Twelve minutes of careful manual work, and a 1-in-6 chance someone fat-fingered a flag or did the hosts in the wrong order and took the service down. Their change-fail rate on deploys was around 18%, almost all of it human error in the runbook.

Their entire deploy, re-expressed as a Deployment, became this: change `image: …@sha256:OLD` to `…@sha256:NEW` in one YAML file, and run one command. The 14 steps collapsed into a one-line diff. The rollout — surge a new pod, wait for readiness, retire an old pod, three times — is performed by the reconcile loop, identically every time, with no host ordering to get wrong. Their deploy change-fail rate dropped from ~18% to under 3% (the residual being genuine bad code, not deploy mechanics), and the deploy went from a 12-minute two-person ceremony to a 90-second one-person `apply`. The *reasoning* behind the win: they moved the deploy logic from a fragile human procedure into a declarative spec that a deterministic controller executes. That is the whole game.

## 4. Service: a stable address in front of moving pods

Pods are ephemeral and their IPs churn. So how does anything ever talk to your three web pods if they keep getting new IPs? The **Service** is the answer. A Service is a stable virtual IP (a "ClusterIP") plus a stable DNS name (`web.shop.svc.cluster.local`, or just `web` from within the same namespace) that load-balances across whichever pods currently match its selector. Callers talk to the Service name; the Service keeps an always-current list of healthy pod IPs behind it and spreads traffic across them. When a pod dies and a new one with a new IP appears, the Service's endpoint list updates automatically — the caller never notices, because the caller was talking to the stable Service name the whole time.

This is the decoupling that makes ephemeral pods usable. A Service is the *contract* — "reach me here" — and the pods are the *fungible* backends behind it. The wiring, again, is labels: the Service's `selector` is a label query, and it dynamically tracks every pod matching that query. There is no list of pod IPs you maintain; the reconcile loop maintains the endpoint list for you, in real time, as pods come and go.

```yaml
apiVersion: v1
kind: Service
metadata:
  name: web
  namespace: shop
spec:
  type: ClusterIP          # stable in-cluster IP; not exposed outside the cluster
  selector:
    app: web               # routes to every pod with label app=web (any ReplicaSet)
  ports:
    - name: http
      port: 80             # the Service port callers hit
      targetPort: 8080     # the containerPort on the pods
```

Notice the selector is `app: web` — the *same label* the Deployment stamps on its pods. This is why labels are the connective tissue of Kubernetes: the Deployment owns pods by label, the Service finds pods by label, the Ingress (next section) targets the Service. Nothing references a pod by name or IP. Everything references by *label query*, and the loops keep the queries live. There is a subtle and useful consequence: because the Service selects by label and not by Deployment, during a rolling update it naturally load-balances across *both* the old and new ReplicaSet's pods (both carry `app: web`), so traffic shifts gracefully as the new pods come up and the old ones go down. The Service does not know or care which version a pod is — it knows the label.

There are three Service types worth knowing, and choosing among them is a real delivery decision:

| Service type | What it gives you | When to reach for it |
| --- | --- | --- |
| `ClusterIP` | Stable IP + DNS reachable only *inside* the cluster | The default and the right answer for service-to-service traffic; pair with an Ingress to reach the outside |
| `NodePort` | Opens the same port on every node's IP | Quick demos, bare-metal without a load balancer; rarely the right production answer |
| `LoadBalancer` | Provisions a cloud load balancer with an external IP | Exposing a single service directly on a cloud; one LB per service can get expensive at scale |

For HTTP services the common production pattern is `ClusterIP` Services behind a single shared Ingress (one external load balancer, many services), rather than a `LoadBalancer` per service. That is the next object.

## 5. Ingress and Gateway API: routing the outside world in

A `ClusterIP` Service is reachable inside the cluster but not from the internet. **Ingress** is the object that defines HTTP(S) routing from outside into your Services: host and path rules ("`app.example.com/api` goes to the `api` Service, everything else goes to `web`"), plus TLS termination (which certificate to present). An Ingress is just a set of routing rules; the actual traffic is handled by an *ingress controller* (nginx-ingress, Traefik, or a cloud controller) running in the cluster, which reads your Ingress objects and configures itself accordingly — the same declarative pattern again, one layer up. You declare the desired routing; a controller makes it real.

```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: web
  namespace: shop
  annotations:
    cert-manager.io/cluster-issuer: letsencrypt-prod   # auto-provision TLS
spec:
  ingressClassName: nginx
  tls:
    - hosts: [app.example.com]
      secretName: web-tls          # the cert lives in a Secret
  rules:
    - host: app.example.com
      http:
        paths:
          - path: /
            pathType: Prefix
            backend:
              service:
                name: web          # routes to the web Service, which load-balances to pods
                port:
                  number: 80
```

The chain end to end, which the section 3 figure draws: an external user hits `https://app.example.com`, the ingress controller terminates TLS and matches the host/path rules, routes to the `web` Service's ClusterIP, and the Service load-balances across the healthy `app: web` pods. Three objects — Ingress, Service, Deployment — and a request reaches your code. That trio *is* a deployable web app.

A note on the future: the newer **Gateway API** is the evolving successor to Ingress, splitting the single Ingress object into a richer set (`GatewayClass`, `Gateway`, `HTTPRoute`) with cleaner separation between the platform team that owns the gateway and the app team that owns the routes, plus first-class support for traffic splitting (useful for canaries). For most delivery work today, Ingress is what you will encounter; if you are starting fresh on a recent cluster, Gateway API is worth a look. The delivery model is identical either way: you declare routing as YAML, a controller reconciles it.

## 6. The reconcile loop's delivery payoff: rollouts and self-healing

Everything so far has been setup for the two behaviors that make Kubernetes a *delivery* platform rather than just a container runner: controlled rollouts and self-healing. Both fall directly out of the declarative model.

![A timeline of a rolling update where applying a new image digest spawns a new ReplicaSet that scales up, a readiness probe gates traffic, old pods scale down, and kubectl rollout status reports done](/imgs/blogs/kubernetes-for-delivery-the-objects-that-matter-3.png)

The timeline above is the rolling update over time: applying the new digest, the new ReplicaSet scaling up, readiness gating traffic, old pods scaling down, and `rollout status` reporting done. And the contrast that makes self-healing concrete — a one-shot imperative deploy that cannot recover from a pod death, versus a declared state the loop keeps true — is the next figure.

![A two-column before and after comparison contrasting an imperative deploy script that runs once and cannot recover from a pod death against a declared desired state where the reconcile loop recreates a failed pod with no human](/imgs/blogs/kubernetes-for-delivery-the-objects-that-matter-4.png)

**The rollout.** When you change the image digest in the Deployment and `kubectl apply`, the Deployment controller creates a new ReplicaSet for the new pod template. Governed by your `strategy` (`maxSurge: 1`, `maxUnavailable: 0`), it brings up one new pod, *waits for its readiness probe to pass*, adds it to the Service's endpoints, then scales the old ReplicaSet down by one — repeating until the new ReplicaSet is at full replicas and the old one is at zero. The deploy is "done" when the loop reports the new ReplicaSet fully available. You observe this, not script it:

```bash
# Ship the new version
kubectl apply -f manifests/

# Watch the rollout converge; this blocks until success or fails on timeout
kubectl rollout status deployment/web -n shop --timeout=120s
# deployment "web" successfully rolled out

# If it goes wrong, roll back to the previous ReplicaSet — declaratively
kubectl rollout undo deployment/web -n shop
```

The crucial delivery property: **readiness gates the rollout.** If a new pod never passes its readiness probe (bad config, missing dependency, crash on boot), it never receives traffic and the old pods are never retired — `rollout status` hangs and then fails on the timeout, leaving the *old, working* version fully in place. The declarative model plus a real readiness probe gives you a deploy that fails *safe* by default. This is also where progressive delivery (canaries, automated analysis, SLO-gated promotion) plugs in — but that is the reliability layer, owned by the SRE side of this content; see `/blog/software-development/site-reliability-engineering/deploying-safely-progressive-delivery` for the why-it's-safe and `/blog/software-development/microservices/deployment-strategies-blue-green-canary-feature-flags` for the fleet-level strategy. Here, the toolchain fact is: the rollout is a reconcile-loop behavior you configure with a strategy and gate with a probe.

**Self-healing.** This is the payoff that genuinely changes your on-call life. Because the Deployment declares "3 replicas" as desired state, the ReplicaSet controller's only job is to make actual replicas equal desired replicas — forever. Kill a pod and the loop makes a new one. Drain a node and the loop reschedules its pods onto other nodes. You did nothing. Contrast that with the imperative `deploy.sh` model in the before/after figure above: the script ran once, the pod died, and there is no one watching — you get paged, you SSH in, you restart it by hand, MTTR is measured in minutes-to-find-out-plus-minutes-to-fix. The declarative model collapses that to seconds with no human in the path.

![A branching graph where a desired count of three replicas and two failure events, a pod crash and a node loss, all feed into the loop observing actual versus desired, which schedules new pods and returns to three with no page](/imgs/blogs/kubernetes-for-delivery-the-objects-that-matter-5.png)

The figure shows the self-healing path: the desired count and any failure event (pod crash, node loss) both feed the observe step, which diffs actual against desired and schedules replacements until it is back to three — and no one was paged. Let me make this concrete enough to try yourself.

#### Worked example: self-healing in action versus a one-shot deploy

Set up the Deployment from section 3 (`replicas: 3`). In one terminal, watch the pods:

```bash
kubectl get pods -n shop -l app=web --watch
# web-7d9c-4klm   1/1   Running
# web-7d9c-9xpq   1/1   Running
# web-7d9c-2vbn   1/1   Running
```

In another, kill one on purpose:

```bash
kubectl delete pod web-7d9c-4klm -n shop
# pod "web-7d9c-4klm" deleted
```

Flip back to the watch terminal. Within a second or two you see the ReplicaSet controller react — the deleted pod goes `Terminating`, and a brand-new pod (`web-7d9c-q7tz`, a new name and a new IP) appears `Pending` then `Running`. You never touched it. The loop saw "desired 3, actual 2" and closed the gap. Now go harder: cordon and drain a node (`kubectl drain node-2 --ignore-daemonsets`). Every pod on that node is evicted and the loop reschedules them onto surviving nodes, all while the Service quietly keeps load-balancing across whichever pods are currently `Ready`. The service stayed up through a node loss with zero human action.

Now contrast that with the imperative alternative the before/after figure two paragraphs up draws exactly: you `docker run` three containers on three hosts. One host reboots. You now have two containers and no controller. The service is degraded until monitoring catches it and pages you, you find the dead host, and you restart the container by hand. Best case that is ten minutes of MTTR; at 3am it is closer to forty. The declarative cluster did the same recovery in two seconds because closing the desired-versus-actual gap is not an emergency procedure there — it is the steady-state behavior of the loop. *That* is why teams move to Kubernetes for delivery: not for the YAML, but for the property that "running" becomes a self-correcting invariant instead of a hope.

The honest stress test, because self-healing has limits. What if the *whole* deployment is bad — the new image crashes on startup? The loop faithfully restarts it into CrashLoopBackOff; it is driving toward a desired state that is itself broken, and readiness gating means the *rollout* fails safe (old version stays up) but a fresh deploy of a broken image will not self-heal into health — only a corrected manifest will. What if you lose *quorum* of nodes, more capacity than the survivors can hold? The loop wants to reschedule but there is nowhere to place pods; they sit `Pending` until capacity returns. Self-healing recovers from *partial* failure against *sufficient* capacity. It is not a substitute for capacity planning or for designing the app to fail gracefully — for that reliability-engineering layer, see `/blog/software-development/site-reliability-engineering/designing-for-failure`. The loop guarantees convergence *if convergence is possible*.

#### Worked example: how long does the rollout actually take, and why image size matters

A team complained their deploys "felt slow" — `rollout status` regularly took 4 to 6 minutes for a service with `replicas: 6`, `maxSurge: 1`, `maxUnavailable: 0`. Let us decompose the time so the fix is obvious instead of guessed.

A rolling update with `maxSurge: 1` swaps pods roughly one at a time, so the rollout duration is approximately the number of replicas times the per-pod turnaround. Per pod, the turnaround is: image pull time + container start time + the readiness wait (the probe's `initialDelaySeconds` plus however many `periodSeconds` cycles until it passes). For this team the image was 1.4 GB, pulling onto nodes that did not already have it took about 35 seconds, container start was ~5 seconds, and the readiness probe had `initialDelaySeconds: 10` plus typically two 5-second cycles before passing — call it 20 seconds. So per pod $\approx 35 + 5 + 20 = 60$ seconds, and across 6 replicas one-at-a-time that is roughly $6 \times 60 = 360$ seconds — six minutes. The arithmetic matches the complaint exactly, and now the levers are visible.

Two changes cut it dramatically. First, shrinking the image with a multi-stage build and a distroless base took it from 1.4 GB to about 90 MB, dropping the cold pull from 35 seconds to ~4 — and on nodes that already cached the base layer, near zero. (That image-size win is the build/Dockerfile sibling posts' territory; it pays off *here*, at deploy time, by shortening every pull in every rollout.) Second, tuning the readiness probe — `initialDelaySeconds: 3` and `periodSeconds: 2` for a service that genuinely warms up in two seconds — cut the readiness wait from ~20 seconds to ~5. New per-pod turnaround: $\approx 4 + 5 + 5 = 14$ seconds; across 6 replicas, about 84 seconds. The deploy went from ~6 minutes to ~1.4 minutes with no change to the application code at all — just a smaller artifact and an honestly-tuned probe. And bumping `maxSurge: 2` (bring up two new pods at a time, acceptable since spare capacity existed) roughly halved it again. The reasoning the team took away: a rollout's wall-clock time is *mechanical* and *measurable* — decompose it into pull + start + readiness per pod, times the surge cadence, and you know exactly which lever to pull instead of declaring deploys "just slow."

## 7. ConfigMap and Secret: externalizing what changes per environment

The build-once principle says one immutable image flows through dev, staging, and prod unchanged. But the *config* differs per environment — the database URL, feature flags, the log level, API keys. If config were baked into the image, you would need a different image per environment and you would have broken build-once. So Kubernetes externalizes config into two objects you inject into pods at runtime: the **ConfigMap** (non-sensitive key-value config) and the **Secret** (the same idea for sensitive values like passwords and tokens). Same image, different ConfigMap/Secret per environment — that is how one artifact stays promotable.

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: web-config
  namespace: shop
data:
  LOG_LEVEL: "info"
  FEATURE_NEW_CHECKOUT: "true"
  UPSTREAM_TIMEOUT_MS: "800"
---
apiVersion: v1
kind: Secret
metadata:
  name: web-secrets
  namespace: shop
type: Opaque
stringData:
  DATABASE_URL: "postgres://web:CHANGED@db.shop.svc:5432/shop"
  SESSION_KEY: "rotate-me"
```

The Deployment's `envFrom` block (section 3) pulls both in as environment variables, or you can mount them as files. A few delivery-relevant truths to internalize:

A bare `Secret` is **not encrypted** — it is only base64-encoded, which is encoding, not security. Anyone with read access to the namespace can decode it, and committing a raw Secret YAML to Git is a leak waiting to happen. The right pattern for delivery is to *not* keep raw secret values in Git at all: use Sealed Secrets (an encrypted Secret that only the cluster can decrypt, safe to commit), the External Secrets Operator (the cluster pulls from Vault or a cloud secret manager at runtime), or OIDC federation for cloud credentials. That whole story is the dedicated sibling `configuration-and-secrets-in-kubernetes`; here, just know that ConfigMap and Secret are *where* config enters the pod, and that a Secret's job is injection, not encryption.

There is one famous footgun: **changing a ConfigMap or Secret does not, by itself, restart the pods.** The pods that already mounted the old values keep running with them — the reconcile loop reconciles the *ConfigMap object*, but the Deployment did not change, so no rollout happens. To pick up new config you must trigger a rollout, e.g. `kubectl rollout restart deployment/web -n shop`, or change something in the pod template (a common trick is to put a checksum of the config in a pod annotation so editing the config changes the template and triggers a roll). This is exactly why the matrix below answers "triggers rollout?" with "only if rolled" for config — a real gotcha that has shipped many a "I updated the config but nothing changed" incident.

## 8. The objects, at a glance: what you edit and what edits itself

Step back and look at the delivery surface as a whole. There are about six object kinds you care about, and the practical questions are: what does each do, do *you* hand-edit it, and does changing it trigger a rollout? That is a comparison, and a comparison wants a table.

![A matrix listing Deployment, Service, Ingress, and ConfigMap plus Secret against columns for what each object does, whether you edit it, and whether changing it triggers a rollout](/imgs/blogs/kubernetes-for-delivery-the-objects-that-matter-6.png)

| Object | What it is for | You hand-edit it? | Changing it triggers a rollout? |
| --- | --- | --- | --- |
| **Pod** | A running instance of your container(s); the unit of *running* | Almost never — a controller creates it | n/a (you do not own it directly) |
| **ReplicaSet** | Keeps N pods matching a template alive | Never — the Deployment creates and versions it | n/a (managed by the Deployment) |
| **Deployment** | Replicas + pod template + update strategy; the unit of *shipping* | Yes — this is the object you change to deploy | **Yes** — bumping the image rolls the pods |
| **Service** | Stable IP + DNS + load-balancing across pods by label | Yes, but rarely changes after creation | No |
| **Ingress / Gateway** | HTTP host/path routing + TLS from outside to a Service | Yes, but rarely changes after creation | No |
| **ConfigMap / Secret** | Externalized config and secrets injected into pods | Yes | Only if you trigger a roll (a known footgun) |
| **Namespace** | Isolation/tenancy boundary for a set of objects | Yes, set once per team/env | No |

The matrix figure above captures the same shape visually. The pattern to take away: **you hand-edit Deployments, Services, Ingresses, ConfigMaps/Secrets, and Namespaces; the platform's controllers create and manage Pods and ReplicaSets for you.** And only the Deployment (via its pod template) and a deliberately-rolled config change actually cause new pods to ship. Everything else is steady-state wiring.

The one object on this list I have not detailed is the **Namespace** — the cheap, essential isolation boundary that scopes names, RBAC, resource quotas, and network policy. You put `shop` and `payments` in separate namespaces so a typo in one cannot touch the other, so you can grant a team access to its namespace without handing it the cluster, and so you can attach a `ResourceQuota` (this namespace may consume at most N CPU and M memory) and a default `NetworkPolicy` (pods here may only talk to a named set of peers) as a blast-radius and security boundary. Names are scoped *within* a namespace, which is why a `Service` named `web` in `shop` and another `web` in `payments` coexist happily, and why intra-cluster DNS uses the fully qualified `web.shop.svc.cluster.local`. For delivery it is mostly a one-line `metadata.namespace` on every object plus a `kubectl create namespace shop` you ran once — but it is also the natural unit you map an *environment* or a *team* onto, and a common GitOps layout is one directory of manifests per namespace, synced independently.

It is worth dwelling for a moment on **labels and selectors**, because they are the single mechanism that makes all of this composition work and they are easy to under-appreciate. A label is just a key-value tag (`app: web`, `tier: frontend`, `version: v2`) you stamp on objects; a selector is a query over labels. Nothing in Kubernetes wires objects together by hardcoded reference — the Service does not list its pods, the Deployment does not list its pods, the Ingress does not list endpoints. They all use *label queries* that the controllers keep continuously satisfied. This is what lets the system be dynamic: when a rolling update creates a new pod carrying `app: web`, the Service's selector matches it the instant it appears and the loop adds it to the endpoint list; when the old pod disappears, the selector stops matching and the loop removes it. You wrote a query once; the loop keeps the answer current forever. Get the labels wrong — a `selector` of `app: web` but a pod template stamping `app: webapp` — and the Deployment will create pods that the Service cannot find, and you will stare at a perfectly healthy set of pods serving zero traffic. Label/selector mismatch is one of the most common "everything looks fine but nothing works" k8s delivery bugs, and `kubectl get endpoints web` (empty list) is how you catch it.

## 9. The manifest is the deliverable: kubectl apply, and what NOT to touch

Here is the synthesis. Your application's *deployment definition* is a set of YAML manifests — Deployment, Service, Ingress, ConfigMap — sitting in a directory in Git. That directory *is* the deliverable, in the same way the image is the deliverable from the build stage. Everything-as-code, the second principle of the series, means this deployment definition is versioned, reviewed in pull requests, and applied by a controller — not clicked together in a console.

![A layered stack showing manifests in Git with a digest-pinned image, reviewed as a pull request diff, applied via kubectl or Argo CD, the cluster converging to the manifest, and Git history serving as the audit log](/imgs/blogs/kubernetes-for-delivery-the-objects-that-matter-7.png)

The figure draws the manifest's path: YAML in Git with the image pinned by digest, the change reviewed as a PR diff (the diff *is* the deploy — you can see exactly what will change), applied via `kubectl apply` or, later, an Argo CD sync, the cluster converging to match, and Git history standing as a complete audit log of every change to production. Notice you could swap `kubectl apply` for a GitOps controller without changing a single manifest — that interchangeability is the declarative model paying off again.

The way you ship matters as much as what you ship. There is a right verb and a wrong one:

```bash
# RIGHT: declarative. Apply the desired state; k8s computes the diff and converges.
# Idempotent — run it twice, same result. Safe to re-run in a pipeline.
kubectl apply -f manifests/ -n shop

# Watch it land
kubectl rollout status deployment/web -n shop --timeout=120s
kubectl get deploy,svc,ing,pods -n shop
kubectl describe deploy/web -n shop          # events, conditions, rollout history

# WRONG (anti-pattern): imperative. Creates objects but they are NOT the
# declared desired state in a file — re-running fails ("already exists"),
# and there is no single source of truth to diff or revert.
kubectl create deployment web --image=ghcr.io/acme/web:latest   # don't
kubectl scale deployment web --replicas=5                       # drift!
kubectl edit deployment web                                     # drift!
```

The distinction is not pedantic. `kubectl apply -f` takes your files as the source of truth and reconciles toward them — it is idempotent, reviewable, and reversible (re-apply the old files to roll back). The imperative commands (`create`, `scale`, `edit`, `patch` done by hand) mutate live cluster state *out of band* from your files, which means your Git no longer matches reality. That gap is **drift**, and drift is how you end up at 3am staring at a production cluster nobody can explain because someone `kubectl edit`-ed it during an incident six weeks ago and never wrote it down. The discipline: the files in Git are the truth; you change the files and apply; you never hand-mutate the live cluster except to *investigate*. (When you graduate to GitOps, the controller *enforces* this — it will actively revert any out-of-band drift back to what Git says. Another reason the declarative model and GitOps are the same idea at two scales.)

![A taxonomy tree splitting Kubernetes for delivery into what you ship as YAML, the Deployment, Service, ConfigMap, and Ingress, versus what the platform owns, the control plane, scheduler, CNI, and nodes](/imgs/blogs/kubernetes-for-delivery-the-objects-that-matter-8.png)

And now the boundary the whole post has been circling, drawn as a taxonomy in the figure above: **what you deploy by hand versus what is platform you never touch.** You ship the workload objects — Deployment, Service, Ingress, ConfigMap, Secret — as YAML. You do *not* deploy the control plane (the API server, the controller manager, the scheduler, etcd), the CNI network plugin, the kubelet on each node, or the nodes themselves. Those are stood up and operated by a cloud provider's managed offering (EKS, GKE, AKS) or by a platform team, and from a delivery standpoint they are a given — the machine that reads your manifests, not something in your pull request. If you find yourself writing scheduler config or CNI manifests as part of *shipping a feature*, you have crossed from delivery into platform engineering, and you should stop and ask whether that belongs to a different team. The delivery surface is deliberately small. Keeping it small is what lets a product engineer ship to Kubernetes without becoming a Kubernetes operator. For how that platform layer actually works — the scheduler's placement logic, pod networking, the architecture of running a fleet — cross-link out to the system-design and microservices content rather than expecting it here; this post is the *what you apply to ship* view by design.

## 10. One manifest set, many environments: templating without losing the plot

You have one immutable image flowing through dev, staging, and prod (build-once), and you have one set of manifests describing the deployment. But the manifests are not *identical* across environments — prod wants 6 replicas and bigger resource limits, staging wants 2 and a different hostname, dev wants 1 and a debug log level. Copy-pasting the manifest three times and hand-editing each copy is a maintenance trap: a change to the Deployment template now has to be made in three places, and they drift apart. So delivery teams reach for a templating or overlay layer. The two dominant tools are **Helm** and **Kustomize**, and choosing between them is a real decision worth understanding rather than cargo-culting.

**Kustomize** (built into `kubectl` as `kubectl apply -k`) takes a different, and to my taste cleaner, approach than templating: it keeps a `base/` set of plain, valid YAML manifests and then layers environment-specific *patches* on top via `overlays/prod`, `overlays/staging`. There are no template placeholders — every file is a real Kubernetes manifest you could apply directly. An overlay says "take the base, and for prod, patch `replicas` to 6 and the image digest to this." Because the base is plain YAML, it stays readable and lintable, and the patches make the *difference per environment* explicit and small.

```yaml
# overlays/prod/kustomization.yaml
apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization
resources:
  - ../../base                 # the plain Deployment + Service + Ingress
namespace: shop
images:
  - name: ghcr.io/acme/web
    digest: sha256:9f2b...c41a   # pin the promoted digest here, per environment
patches:
  - target: { kind: Deployment, name: web }
    patch: |-
      - op: replace
        path: /spec/replicas
        value: 6                 # prod runs 6; base/staging differ
configMapGenerator:
  - name: web-config
    literals:
      - LOG_LEVEL=info
```

**Helm** instead treats your manifests as a *templated chart*: files full of `{{ .Values.replicas }}` placeholders, rendered against a `values.yaml` (and per-environment `values-prod.yaml`) to produce the final YAML. Helm adds packaging (a versioned chart you can publish and share), a release concept with history and `helm rollback`, and a large ecosystem of off-the-shelf charts for third-party software (a database, a monitoring stack). The cost is that your manifests are no longer plain YAML — they are Go-template source that you cannot read as Kubernetes objects until rendered, and complex charts become genuinely hard to debug.

| Concern | Kustomize | Helm |
| --- | --- | --- |
| Mechanism | Patch/overlay plain YAML bases | Render Go-template placeholders against values |
| Are your files valid YAML? | Yes — every file applies directly | No — templated source until rendered |
| Packaging/sharing | None built in | Versioned, publishable charts; rich ecosystem |
| Release history/rollback | Use Git + `apply` | Built-in `helm rollback` + release revisions |
| Sweet spot | Your own services, a few environments | Distributing software, heavy parameterization |
| Failure mode | Overlays sprawl into many tiny patches | Template logic becomes unreadable spaghetti |

The pragmatic guidance: for shipping *your own* handful of services across a few environments, Kustomize's plain-YAML-plus-overlays is simpler and keeps the manifest readable as actual Kubernetes objects — start there. Reach for Helm when you are *distributing* software others will install (a chart is a great package format) or when parameterization is genuinely heavy. Many teams end up using both: Helm for third-party charts they consume, Kustomize for their own services. Either way the principle is unchanged — one image, one base definition, small explicit per-environment differences, all in Git. Templating is its own deeper sibling topic in this series; here the point is just that it sits *on top of* the same objects you have already learned, and it does not change the declarative model one bit. The output of Helm or Kustomize is still the same Deployment, Service, and Ingress, applied by the same reconcile loop.

## 11. Stress-testing the model: what happens when things go sideways

A principle is only trustworthy once you have pushed on its edges. Here are the failure scenarios that come up in real delivery, and how the declarative model behaves under each — including where it does *not* save you.

**What if two pull requests merge at once and both touch the same Deployment?** The Kubernetes API uses optimistic concurrency: every object has a `resourceVersion`, and an update that was computed against a stale version is rejected with a conflict. With `kubectl apply` and server-side apply, the second writer's change is merged field-by-field against the latest object rather than blindly overwriting, and genuine conflicts on the *same* field surface as errors rather than silent last-write-wins. The practical safety, though, is upstream: if both PRs went through review and CI before merging to the branch a controller applies, the merge conflict is resolved in Git (the source of truth) before it ever reaches the cluster. This is another argument for the Git-is-truth discipline — Git's merge machinery is a far better place to resolve two simultaneous changes than the live cluster.

**What if the registry is down mid-deploy?** A rolling update needs to pull the new image onto the nodes scheduling the new ReplicaSet's pods. If the registry is unreachable, those pods sit in `ImagePullBackOff` — and here the model's fail-safe design pays off: because `maxUnavailable: 0`, the old pods are *not* retired until the new ones are ready, so the registry outage stalls the rollout but does **not** take down the running service. `rollout status` will eventually time out and report the failure, and you still have the old version fully serving. The lesson for your manifests: a conservative `maxUnavailable: 0`/`maxSurge: 1` strategy turns a mid-deploy infrastructure hiccup into a stalled (recoverable) rollout instead of an outage. A mitigation worth knowing: pulling by *digest* (which you do) plus image pre-pull or a pull-through cache reduces exposure to registry blips.

**What if the readiness probe is flaky or the metric is noisy?** A readiness probe that intermittently fails will flap a pod in and out of the Service endpoints, causing intermittent errors, and during a rollout a flaky probe can stall the roll indefinitely (the loop never sees the new pod as durably ready). The fix is probe hygiene — a cheap, dedicated `/healthz` that checks only liveness-relevant local state, sane `periodSeconds` and `failureThreshold`, and *not* having the readiness probe call downstream dependencies (or one slow dependency takes your whole fleet out of rotation). For the genuinely hard case of gating a deploy on a *noisy SLI*, that is the domain of progressive-delivery analysis (Argo Rollouts / Flagger), which uses statistical analysis windows rather than a single binary probe — cross-link `/blog/software-development/site-reliability-engineering/deploying-safely-progressive-delivery`.

**What if the rollback also fails?** `kubectl rollout undo` re-applies the previous ReplicaSet's template — but if that previous version *also* cannot start (a shared dependency broke, a database migration already ran forward), rolling back the deployment alone will not save you. This is why "roll back" is not a universal mitigation: the most dangerous deploys are the ones coupled to forward-only state changes (schema migrations). The discipline is to decouple them — ship backward-compatible schema changes ahead of the code that needs them, so the previous code version is always runnable against the current schema. That is a database-delivery concern; see `/blog/software-development/database/zero-downtime-schema-migrations`. The k8s-level takeaway: the reconcile loop makes *deploy* rollback trivial, but it cannot roll back state, so design state changes to never require it.

**What if the loop and you fight over the same field?** If you run a HorizontalPodAutoscaler (which writes `replicas`) *and* you hardcode `replicas: 3` in a manifest you keep applying, the two controllers will tug-of-war and your pods will flap. The resolution is to not declare a field that another controller owns — omit `replicas` from the manifest when an HPA manages it, and let the autoscaler be the sole writer. This generalizes: in the declarative world, every field should have exactly *one* owner. Two writers of one field is the drift problem in miniature, and it is exactly what server-side apply's field-ownership tracking exists to surface.

The through-line across all five: the declarative model is fail-safe *by design* for the failures it can express (a stalled rollout leaves the old version up; a dead pod is recreated; a conflict is rejected, not silently lost), and it is honest about the failures it cannot fix (broken state, a wrong declaration, a flaky signal). Knowing which is which is the difference between trusting the model and being surprised by it.

## 12. War story: the deploy script that could not heal, and the edit that caused the drift

Two real-shaped stories, because the abstract case lands harder when it has a body.

**The script that ran once.** A payments-adjacent team I consulted with ran a fleet of services with a beautifully engineered Bash deploy system — Ansible playbooks, rolling SSH, health checks, the works. It worked great *during a deploy*. The problem was everything that happened *between* deploys. A kernel bug caused one service's process to wedge roughly twice a week, always in the small hours. Their playbook had no idea — it had finished running days ago. So the rotation got paged, SSH'd in, restarted the process, and went back to bed, twice a week, for months. They had built an excellent imperative deploy and zero self-healing, because in the imperative world those are separate problems you must each solve. When they moved the workload to a Deployment, the *exact same* wedge still happened — but now the liveness probe failed, the kubelet restarted the container, and the rotation slept through it. They did not write any new recovery logic. Self-healing came free with the declarative model, because "keep it running" stopped being a procedure and became an invariant the loop enforces. Their deploy-related pages dropped to roughly zero.

**The edit that broke an upgrade.** A different team, six months into Kubernetes, hit a baffling incident: a routine `kubectl apply` of a new version *reduced* their replica count from 12 back to 3 in the middle of a traffic spike, browning out the service. The root cause was drift. Weeks earlier, during a load event, an engineer had run `kubectl scale deployment api --replicas=12` to add capacity fast — imperatively, live, never reflected in the Git manifest, which still said `replicas: 3`. It worked, the incident passed, everyone forgot. Then the next normal deploy ran `kubectl apply -f` from Git, which faithfully reconciled the cluster toward the *declared* desired state — `replicas: 3` — and scaled them down by nine right when they were needed. The cluster did exactly what it was told; the *declaration* was just stale because someone had changed reality out of band. The fix was procedural, not technical: the manifest in Git is the only source of truth, scaling changes go through a PR (or an autoscaler, which writes a different field), and `kubectl scale`/`edit` are for breaking glass during an incident *and then immediately reconciling Git to match*. The lesson generalizes to the whole declarative model: its power — that the loop enforces what is declared — is also its sharpest edge, because it will faithfully enforce a *wrong* declaration just as diligently as a right one. The discipline that keeps you safe is making sure the declaration in Git always matches the reality you intend.

Both stories have the same moral from two directions. The declarative model gives you self-healing and reproducible deploys *for free* — but only if the declared state is the *true* state. Treat Git as truth, never hand-mutate the cluster except to investigate, and the model is a gift. Let drift creep in and the model will dutifully enforce your forgotten mistakes.

## 13. How to reach for this (and when not to)

Kubernetes is a powerful delivery target and an enormous operational liability if you adopt it for the wrong reasons. Some honest guidance.

**Reach for Kubernetes as a delivery target when** you are running multiple long-lived services that need self-healing, rolling updates, horizontal scaling, and a uniform deploy interface across a team or fleet — and when *someone else operates the control plane for you* (a managed offering like EKS/GKE/AKS, or a dedicated platform team). The declarative model genuinely pays off at the point where "keep these services running and let many engineers ship to them safely" is a real, recurring problem. If you have a handful of services and a platform team or cloud handling the cluster, the delivery surface in this post — five objects and `kubectl apply` — is a small, learnable, high-leverage skill.

**Do not reach for Kubernetes when** you have one app and three engineers and no platform team. Standing up and operating a cluster yourself is a full-time job that produces zero customer value; for a small team a PaaS (Fly.io, Render, Railway, Cloud Run, App Runner, Heroku-likes) gives you the same declarative-ish "I want N instances of this image" deploy with none of the operational tax, and you can graduate to Kubernetes later when the fleet justifies it. "We might need to scale someday" is not a reason to take on a distributed operating system today. Similarly, do not run your *own* control plane unless operating Kubernetes is itself your product — let the cloud do it.

**Do not over-build the manifests, either.** You do not need a Service mesh, custom controllers, or admission webhooks to ship a web app. Start with a Deployment, a Service, an Ingress, and a ConfigMap — the five objects here — and add complexity only when a concrete need forces it. The templating layer (Helm, Kustomize) becomes worth it when you have *multiple environments or services* sharing structure; for a single app one set of plain YAML files is fine and clearer. (Templating is its own sibling topic in this series.) And do not adopt GitOps tooling before you can comfortably `kubectl apply` by hand — learn the declarative model first; automate the apply second.

The decision is really about *who operates the platform* and *how many things you ship*. If the answers are "the cloud" and "many," Kubernetes for delivery is excellent and this post is your starting kit. If the answers are "us, by hand" and "one," reach for a PaaS and come back when the shape of the problem changes.

If you have decided Kubernetes is right, the *order* in which you adopt it matters as much as the decision. The path that consistently works: first, write plain Deployment-plus-Service-plus-Ingress YAML and ship it with `kubectl apply` and `rollout status` by hand until the declarative model is second nature and you can read an event stream and a `describe` output fluently. Second, externalize config into ConfigMaps and Secrets and pin images by digest, so one artifact promotes cleanly. Third, add a templating or overlay layer (Kustomize first; Helm if you need packaging) once you have more than one environment. Fourth — and only now — move the `apply` into a pipeline, and finally graduate to pull-based GitOps so Git enforces desired state and drift self-corrects. Skipping ahead — reaching for GitOps and a service mesh before you can hand-deploy a Pod — is how teams end up with an unmaintainable platform nobody on the team actually understands. Each layer in that sequence is only worth its complexity once the layer beneath it is boring. Master the five objects and the one idea first; everything else in this series builds on exactly that foundation.

## 14. Key takeaways

- **One idea carries the whole platform: declare desired state, and a reconcile loop continuously drives actual state to match it.** You do not run deploy scripts; you change a declaration and the loop converges. Self-healing and rolling updates both fall directly out of this.
- **You ship a Deployment.** It declares replicas, the pod template (with a digest-pinned image), and an update strategy; bumping the image and re-applying triggers a rolling update performed by the loop, gated by the readiness probe so it fails safe.
- **Pods are the unit of running, not shipping** — ephemeral, IP-churning, and created by controllers, not by you. Never reference a pod by IP; that is what the Service exists to prevent.
- **Labels and selectors are the connective tissue.** The Deployment owns pods by label, the Service finds them by label, the Ingress targets the Service. Nothing references anything by name or IP, which is exactly what lets pods come and go freely.
- **Externalize config into ConfigMaps and Secrets** so one immutable image stays promotable across environments. A Secret is encoding, not encryption — keep raw values out of Git. Editing a ConfigMap does not restart pods; you must trigger a roll.
- **`kubectl apply -f` (declarative) is the right verb; `kubectl create`/`scale`/`edit` by hand is the anti-pattern** that causes drift. The files in Git are the source of truth; the cluster reconciles toward them; you never hand-mutate the live cluster except to investigate.
- **Know the boundary.** You deploy workload objects — Deployment, Service, Ingress, ConfigMap, Secret, Namespace. You do *not* deploy the control plane, scheduler, CNI, or nodes; that is platform a cloud or platform team operates. Keeping the delivery surface small is what makes shipping to k8s learnable.
- **The declarative model is exactly why GitOps works** — Git becomes the desired-state store and a controller reconciles continuously. Learn the manifest-and-apply skill first; the GitOps automation is the same idea at a larger scale.
- **Measure the payoff honestly:** deploy as a one-line image-digit diff plus one `apply` (not a 14-step runbook), change-fail rate dropping as human deploy error vanishes, and self-healing collapsing MTTR for the common "a pod died" failure from tens of minutes to seconds.

## Further reading

- **Within this series — start here and go sideways:** the intro map [`from-commit-to-production-the-cicd-mental-model`](/blog/software-development/ci-cd/from-commit-to-production-the-cicd-mental-model) for the whole `commit → build → test → package → deploy → operate` spine and the two principles; [`containers-from-first-principles-for-delivery`](/blog/software-development/ci-cd/containers-from-first-principles-for-delivery) for what the image you are deploying actually is; and [`build-once-promote-everywhere-artifacts-and-versioning`](/blog/software-development/ci-cd/build-once-promote-everywhere-artifacts-and-versioning) for why the digest-pinned image matters.
- **Coming in this series (the natural next reads):** `rolling-updates-and-zero-downtime-deploys` (the rollout mechanics this post only sketched), `configuration-and-secrets-in-kubernetes` (the safe way to handle the ConfigMap/Secret layer), and `gitops-git-as-the-source-of-truth` (the declarative model automated — Git as the desired-state store).
- **The reliability layer (cross-link OUT, do not duplicate):** [`deploying-safely-progressive-delivery`](/blog/software-development/site-reliability-engineering/deploying-safely-progressive-delivery) for canaries as an SLO-gated practice, and [`designing-for-failure`](/blog/software-development/site-reliability-engineering/designing-for-failure) for why self-healing is necessary but not sufficient.
- **The fleet layer:** [`deployment-strategies-blue-green-canary-feature-flags`](/blog/software-development/microservices/deployment-strategies-blue-green-canary-feature-flags) for choosing a deploy strategy across many services.
- **Canonical primary sources:** the Kubernetes documentation's Concepts section (Workloads, Services, Ingress, ConfigMaps and Secrets) and the "Kubernetes Object Management" page (the `kubectl apply` declarative model); the original Borg/Omega/Kubernetes lineage paper "Borg, Omega, and Kubernetes" (Burns et al.) for *why* reconciliation-based control was the design choice; and the Twelve-Factor App's config factor for the externalize-config discipline that ConfigMaps implement.
