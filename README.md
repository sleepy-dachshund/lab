# lab
various interests. laboratory setting, not production. typos.

disclaimer: no research per se in here. haven't found a satisfying way to share that work in GitHub yet. open to suggestions. this is all tangential though.

## about me

i'm a quantitative researcher in equity l/s, formerly worked as a lab research scientist. 

## table of contents

__data_pipeline__:

-- __data_gen__ -- simple synthetic data generator. meant to replicate what you might see from a third party vendor. outputs a directory of .csv and .zip files with simulated data. run once and you have some synth data to mess with.

-- __data_loader__ -- the kind of code you'd build to import & process the synth data from data_gen.py, appending new daily data to an existing local file.

__gloves__ -- a super thin package that one could import to help with grabbing basic price/volume data. this is not recommended for serious research, it's yfinance. but it's useful for basic stuff.

__optimization__ -- some different types of portfolio optimization. simple MVO and Black-Litterman (both single-period) with derivations and explanations.

__regression__ -- some notes on regressions and examples.

__other__ -- various other things, options & Black Scholes derivation, etc.
