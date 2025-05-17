Primary:
- Bin the fire data for both dts and sts, modify the various dist functions to optionally support binned data, write a test file to ensure consistency
- Add support for mean(fire_return_distribution) as this should be equivalent to the fire return interval (make sure others agree)
- Once Fred has added the fire-less pixel data, write code to add that to the sts bins
- Email proposed group configuration

Secondary:
- Add support for 1-2 new distributions, possibly update Weibull to be consistent
    - consider moving more common code to a class they inherit from if it will save duplication
- Continue updating statistical test scripts
    - The tests should output to their own sub folders for cleanliness
    - Validate mean of HLH
- Bootstrapping for confidence intervals





Requests of Fred:
- get # of pixels non burnt by eco 1/2 and nlcd
- aspect per pixel joined to datasets
- 

