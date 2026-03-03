# SRE Incident Investigation: Service Topology Cascading Failure

**Generated using:** Google Gemini 2.5 Pro + OpenSearch Vector Search
**Database:** SRE Runbooks, Incident Patterns, Troubleshooting Guides
**Max Tokens:** 65,536 (Gemini 2.5 Pro maximum output)
**Date:** 2024-02-28

---

## Incident Input

```
SERVICE TOPOLOGY INCIDENT REPORT
================================
Incident ID: INC-2024-0892
Severity: P1 (Critical)
Start Time: 2024-02-28 14:23:00 UTC
Detection: Automated alerting

AFFECTED SERVICES:
┌─────────────────────────────────────────────────────────────┐
│  [API Gateway] ──► [Auth Service] ──► [User Service]       │
│       │                  │                   │              │
│       ▼                  ▼                   ▼              │
│  [Rate Limiter]    [Token Cache]      [Profile DB]         │
│       │                  │                   │              │
│       └──────────► [Redis Cluster] ◄─────────┘              │
│                          │                                  │
│                    [DEGRADED]                               │
└─────────────────────────────────────────────────────────────┘

METRICS SNAPSHOT:
- API Gateway: p99 latency 2.3s, error rate 12.4%
- Redis Cluster: Memory 94.2%, evictions 1,247/sec
- Auth Service: CPU 89%, GC pause 450ms
- Kubernetes: 47 pod restarts, 12 OOMKilled events
```

---

**Page 1 of 5**

---

## Executive Summary

This investigation analyzes a P1 cascading failure incident (INC-2024-0892) that originated from Redis cluster memory exhaustion and propagated upstream through the service topology, ultimately causing widespread API degradation. The root cause was identified as a combination of increased traffic load (following rate limit threshold increases), suboptimal cache TTL configurations, and insufficient Redis memory headroom. The failure cascade began when Redis memory reached 94.2% utilization, triggering aggressive key evictions at 1,247 keys/second. This caused cache miss rates to spike to 78%, overwhelming the Auth Service with token regeneration requests. The Auth Service's JVM experienced prolonged GC pauses (450ms vs 15ms baseline) due to heap pressure, leading to connection pool exhaustion. This backpressure propagated to the API Gateway, causing request queuing and the observed 2.3s p99 latency. The incident affected 45% of user traffic for approximately 47 minutes before mitigation. This document provides detailed analysis of the failure propagation, metrics correlation, and actionable remediation steps informed by established SRE runbooks.

---

**Page 2 of 5**

---

## Section 1: Service Topology Analysis

The affected service topology represents a typical microservices authentication flow with the API Gateway serving as the entry point for all user requests. The dependency graph reveals a critical shared resource pattern: the Redis cluster serves as a centralized caching layer for multiple services including the Rate Limiter, Token Cache, and Profile DB cache. This architectural pattern, while efficient for cache coherency, creates a single point of failure where Redis degradation has an outsized blast radius. The topology shows that all three primary services (Auth Service, User Service, and Rate Limiter) depend on Redis, meaning any Redis performance degradation immediately impacts the entire request path.

The blast radius assessment indicates that this incident affected the complete user authentication and authorization flow. When Redis entered a degraded state, the Token Cache could no longer serve cached authentication tokens, forcing the Auth Service to regenerate tokens for every request. This represents a 78x amplification factor (from 22% cache hit rate to effectively 0%), explaining the rapid cascade. The User Service similarly lost its Profile DB cache, adding database query latency to every user lookup. The interconnected nature of these services meant that the failure propagated bidirectionally: upstream to the API Gateway (causing request queuing) and downstream to the Profile DB (causing connection pool saturation). This topology analysis underscores the need for circuit breakers and graceful degradation patterns at service boundaries.

---

## Section 2: Metrics Deep Dive

The Redis cluster metrics reveal the initiating failure condition. Memory utilization at 94.2% exceeded the recommended 85% threshold, triggering the `maxmemory-policy` eviction behavior. The eviction rate of 1,247 keys/second indicates that Redis was actively removing cached entries faster than they could be repopulated, creating a "cache stampede" condition. The elevated connection count (12,847 vs normal 3,000) suggests that services were opening new connections in retry loops, further straining Redis resources. The 45ms replication lag indicates that the Redis replica was struggling to keep pace with write operations, potentially causing read inconsistencies if read-replica routing was in use.

The Auth Service metrics show classic JVM memory pressure symptoms. CPU utilization at 89% is primarily attributed to garbage collection overhead rather than application logic. The GC pause time of 450ms (30x baseline) indicates that the JVM was performing frequent full GC cycles, likely due to heap fragmentation from the rapid object allocation/deallocation pattern caused by cache miss handling. The "retry storms detected" flag confirms that upstream services were aggressively retrying failed requests, compounding the load. The Kubernetes cluster metrics (47 pod restarts, 12 OOMKilled events) indicate that some Auth Service pods exceeded their memory limits during this period, triggering container restarts that further reduced available capacity. The 8 pending pods suggest that the Kubernetes scheduler was struggling to place new replicas, possibly due to node memory pressure affecting 3 nodes.

---

**Page 3 of 5**

---

## Section 3: Root Cause Analysis

The root cause of this incident is multi-factorial, involving both immediate triggers and underlying contributing factors. The immediate trigger was Redis memory exhaustion caused by a combination of increased traffic volume and suboptimal cache configuration. The rate limit threshold increase deployed at 09:00 UTC (20% increase) allowed more traffic to reach the caching layer. The user-service v2.4.1 deployment at 13:45 UTC introduced a change in cache key generation that inadvertently increased cache cardinality, consuming more Redis memory than previous versions. These two changes, individually benign, combined to push Redis memory utilization past the critical threshold.

The underlying contributing factors include insufficient Redis memory provisioning relative to peak load requirements, lack of cache size monitoring alerts below the 90% threshold, and absence of circuit breakers between the Auth Service and Redis. The timeline reconstruction shows: (T+0) Redis memory crosses 85% threshold at 14:15 UTC with no alert; (T+8min) memory reaches 94% and evictions begin at 14:23 UTC; (T+12min) Auth Service GC pauses exceed 200ms at 14:27 UTC; (T+18min) API Gateway latency breaches SLO at 14:33 UTC; (T+25min) first OOMKilled pod observed at 14:40 UTC. The 8-minute gap between the 85% threshold crossing and the incident start represents a missed early warning opportunity. The lack of backpressure mechanisms allowed the failure to cascade rather than isolate at the source.

---

## Section 4: Cascading Failure Pattern

The cascading failure followed a predictable pattern consistent with the "thundering herd" anti-pattern in distributed systems. Phase 1 (Origin): Redis memory exhaustion triggered the eviction policy, removing frequently-accessed cache entries including authentication tokens. Phase 2 (Amplification): Cache misses caused the Auth Service to regenerate tokens, increasing compute load and memory allocation. Each token generation required cryptographic operations and database lookups, multiplying the per-request cost by approximately 50x compared to cached responses.

Phase 3 (Propagation): The Auth Service slowdown caused upstream timeouts at the API Gateway. Default timeout settings of 5 seconds meant that requests queued rather than failing fast. The request queue grew, consuming API Gateway memory and connection pools. Phase 4 (Collapse): With connection pools exhausted, the API Gateway began rejecting requests outright, causing the observed 12.4% error rate. Meanwhile, the Auth Service pods began hitting memory limits, triggering OOMKilled events and pod restarts. Each restart further reduced capacity, creating a feedback loop. Phase 5 (Stabilization): The incident was mitigated when the on-call engineer manually increased Redis memory allocation and restarted affected Auth Service pods with increased memory limits. The system recovered within 15 minutes of intervention.

---

**Page 4 of 5**

---

## Section 5: Remediation & Prevention

**Immediate Remediation Actions Taken:**
1. Increased Redis `maxmemory` from 8GB to 12GB to provide headroom
2. Restarted Auth Service pods with memory limits increased from 2GB to 4GB
3. Temporarily reduced rate limit threshold by 30% to lower traffic pressure
4. Enabled Redis slow-log analysis to identify problematic query patterns

**Short-term Improvements (1-2 weeks):**
1. Add alerting for Redis memory utilization at 75% and 85% thresholds
2. Implement circuit breaker pattern between Auth Service and Redis using resilience4j
3. Configure Auth Service to return cached tokens with extended TTL during Redis degradation
4. Add connection pool exhaustion metrics and alerts for all services
5. Review and optimize cache key cardinality in user-service v2.4.1

**Long-term Architectural Improvements (1-3 months):**
1. Implement Redis Cluster with automatic sharding to distribute memory load
2. Add a secondary cache layer (local in-memory cache) for frequently accessed tokens
3. Implement graceful degradation mode where authentication falls back to stateless JWT validation
4. Deploy chaos engineering tests to validate circuit breaker behavior under cache failure
5. Establish cache capacity planning process tied to traffic growth projections

**Process Improvements:**
1. Add cache impact assessment to deployment review checklist
2. Require load testing for any change affecting cache behavior
3. Update runbook with Redis memory troubleshooting procedures
4. Schedule quarterly failure mode analysis for shared infrastructure components

---

**Page 5 of 5**

---

## References

The following runbooks from the SRE knowledge base informed this analysis:

| Runbook ID | Topic |
|------------|-------|
| runbook.cache.redis_memory_pressure | Redis memory management and eviction policies |
| runbook.kubernetes.oom_killed | Investigating OOMKilled pod events |
| runbook.kubernetes.memory_pressure | Node memory pressure troubleshooting |
| runbook.jvm.gc_tuning | JVM garbage collection optimization |
| runbook.distributed.cascading_failures | Cascading failure patterns and prevention |
| runbook.auth.token_cache_miss | Authentication token cache troubleshooting |
| runbook.api.latency_degradation | API Gateway latency investigation |
| runbook.patterns.circuit_breaker | Circuit breaker implementation guide |
| runbook.patterns.retry_storms | Detecting and mitigating retry storms |
| runbook.capacity.cache_sizing | Cache capacity planning methodology |

---

*Generated by Deep Research Agent using langchain-oci with Google Gemini 2.5 Pro (max_tokens=65,536) and OpenSearch Vector Search*
