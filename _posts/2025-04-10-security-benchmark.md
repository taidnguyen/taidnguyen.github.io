---
layout: blog
categories: blog
published: false
title: "Building a Security-Aware Benchmark for Coding Agents"
description: "Turning Django's CVE history into an evaluation suite that tests whether AI coding agents can fix vulnerabilities without introducing new ones."
date: 2025-04-10
read_time: 5
---

Most coding agent benchmarks measure one thing: can the agent make tests pass? [SWE-bench](https://www.swebench.com/) pioneered this format, giving agents a GitHub issue and checking whether their patch passes the repository's test suite.

But passing tests is not the same as writing secure code. An agent can resolve an issue while introducing a SQL injection, a path traversal, or an authentication bypass, and the benchmark would score it as a success. This gap motivated me to build a benchmark that evaluates both correctness and security awareness.

## The idea

Django has 87 documented CVEs spanning over a decade of development. Each CVE has a fix commit with a clear parent (the vulnerable state), a description of the vulnerability, and regression tests that verify the fix. This gives us a natural evaluation framework:

1. Check out the vulnerable commit
2. Ask the agent to fix it
3. Apply the regression tests from the real fix
4. Score the agent's patch

The key insight is that Django's own security team wrote both the fix and the test. We use the test as an oracle without ever showing the agent the fix itself.

## Four task types

Rather than just "fix this CVE," I designed four task types that test different security capabilities:

**Vulnerability Patch.** The agent gets a description of the vulnerability and must produce a fix. This is the most direct test: can you patch a known security bug?

**Hardening.** The agent is told a vulnerability exists in a specific module but gets no detailed description. It has to find and fix the issue. This tests whether agents can identify vulnerabilities from code alone.

**Localization (Code).** The agent is given the vulnerable file and must identify what's wrong. This tests security reasoning without requiring a full fix.

**Localization (Diff).** The agent sees the actual fix diff and must explain the vulnerability it addresses. This tests whether agents understand security patches.

## What I found

The benchmark is not saturated. Across three model families (GPT-4o-mini, GPT-5-mini, GPT-5) and two prompting strategies (zero-shot and two-shot), the best configuration scored 0.395 overall. Localization tasks were easier than patching tasks, and larger context windows helped significantly.

The most interesting failure mode: agents would exhaust their context window by reading large Django source files in the first few steps, leaving no room to actually reason about the vulnerability. Context management turned out to be as important as security knowledge.

## Scaling beyond Django

The approach generalizes beyond CVEs:

- **Static analysis diffing.** Run a SAST tool (Semgrep, Bandit, CodeQL) on consecutive commits. When findings disappear between commits, that commit likely fixed a security issue. This works on any repository, no CVE metadata needed.
- **Synthetic injection.** Programmatically inject known vulnerability patterns (from OWASP) into well-tested codebases and use the existing test suite as an oracle. This is the most scalable approach since it requires no historical security data at all.
- **Dependency and config tasks.** Commits that bump dependencies for known vulnerabilities, tighten permissions, or add authentication checks are another source of security-relevant evaluation tasks.

The code and full results are available on [GitHub](https://github.com/taidnguyen).
