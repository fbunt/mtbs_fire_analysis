Primary:
- Check the bin code by doing a test with and without
    - Real data? or round generated data to nearest 0.1?
- Once Fred has added the fire-less pixel data, write code to add that to the sts bins
- Continue updating statistical test scripts
    - The tests should output to their own sub folders for cleanliness
    - Validate mean of HLH
- Bootstrapping for confidence intervals
    - Use overlapping fire areas as the bootsrapping unit
    - Run this idea by: John/Downs/???
    - Report

Secondary:
- Add support for 1-2 new distributions, possibly update Weibull to be consistent
    - consider moving more common code to a class they inherit from if it will save duplication

- Plotting
    - Per subset plots
        - Add fire hazard return interval (at 0.8?)
        - Add table of summary statistics (FRI, FHRI)
        - Increase number of bins to an argument eg. 75
        - fix number of decimal places of reported param values
        - Each table should be it's own vertical block
        - each table should include confidence intervals, potentially plotted?
    - Summary table(s)
        - Fitted params, confidence intervals
        - Summary stats, confidence intervals
        - num unique fires, num fire overlaps, num fire pixels


Requests of Fred:
- get # of pixels non burnt by eco 1/2 and nlcd
- aspect/slope/altitude per pixel joined to datasets
- nlcd is u8 datatype, eco is i32 (doesn't really matter but bit inconsistent)


