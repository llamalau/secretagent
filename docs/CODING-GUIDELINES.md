# Coding Guidelines

## Keep it simple and tight

We want to keep the codebase **small** and **understandable**. Some
concrete principles are below.

 * Following existing code conventions where you can. 
 * Before extending "core" (src/secretagent/**, understand what's
   there, and discuss the proposed changes with William.
 * Everything in core should be general
 * If something is general (e.g., retries, self-consistency) 
   consider putting it in core.
 * Be alert to ways to refactor code and remove redundancies.

## Dogfood our system

 * If you are doing stuff that requires using an LLM implement it as
   an Interface/Implementation pair, if you can.  If you can't figure
   out why.

## Provenance everywhere

Every experiment, including ones that rely on a learning optimization
strategy, should be completely tracked so it can be reproduced.  For
instance, when an inplementation is learned with a `learn.Learner` is
is saved along with pointers to the recordings used to train it.

 * Use `config` for everything you can - configs are easy to save and
track.
 * Use `savefile` utilities when you can (these are not just for
  results)
 * Run experiments on a clean `main` branch (so the recorded dates
will determine the codebase used)
 * Add provenance information to your data - tag it as 'backstory'
   somehow.

## Using configurations in code

 * Functions generally access the global config, rather than passing
   down pieces of it as arguments.
 * Fail early when required parameters are missing: When a
   configuration parameter is needed by a subroutine, the caller
   should access that param with 'config.require' and pass down the
   required values as a parameter.
  * For tests or demos that rely on configuration settings, don't
    modify the global config.  Instead use the `with configuration`
    context manager.
