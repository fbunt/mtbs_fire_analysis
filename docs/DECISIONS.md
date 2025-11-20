Decisions and open questions

    This document aggregates known decisions, heuristics, and concerns. Seeded from 10ThingsIHateAboutThisProject.md.

    Source excerpts
    ```markdown
I don't even know what the better solution to these problems are:
 - For Survival times where we we have no fires at a pixel at all, we allocated nlcd to 2003 because it's in the middle of the dataset
 - For Fire intervals, we allocated the nlcd of the interval to the nlcd at the time of the first fire
 - For initial survival times we used the nlcd of the first fire
 - For final survival times we use the nlcd of the last fire

There are a lot of event histories with <10 pixels. 200k of them. Spot checking them and they seem legit, but I've looked at less than 10 of those...
- some of them are just that there's a small area that intersects a large number of fires.
- Some are that the nlcd history changes slightly different per pixel
- Some are that there's a single wetland nlcd pixel within a fire
- etc.

So currently seems fine, but we might want to develop a process to quickly inspect things in QGIS, so that we can double check 50 of them


Need to Merge HLH dist implementation back into distributions. Ideally also:
    Add a registry for the distribution implementations
    Add interfaces for the general distribution class, and a sub-class for scipy distributions to enable quickly adding other distributions that are already in scipy
    Later:
        Consider how things will be abstracted when we want to add covariates, is it possible with proportional hazard, or accelerated failure time or accelerated hazard to have the distribution simply take some transformation arguments so the covariates and their parameters can be abstracted from the distributions?
```

    Next actions
    - Document NLCD assignment heuristics and evaluate alternatives
    - Define QGIS auditing workflow for small event histories (&lt;10 pixels)
    - Plan HLH distribution refactor: registry + base class + SciPy-backed subclass
    - Capture further decisions as they arise in development
