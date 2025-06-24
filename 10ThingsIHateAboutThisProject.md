
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


