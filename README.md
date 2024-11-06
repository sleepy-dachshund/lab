# lab
various interests. laboratory setting, not production. typos.

## about me

i'm a quantitative researcher in equity l/s, formerly worked as a lab research scientist. 

## table of contents

data_gen -- simple synthetic data generator. meant to replicate what you might see from a third party vendor. outputs a directory of .csv and .zip files with simulated data. run once and you have some synth data to mess with.

data_loader -- the kind of code you'd build to import, process the synth data above, appending new daily data to an existing local file.

gloves -- a super thin package that one could import to help with grabbing basic price/volume data. this is not recommended for serious research, it's yfinance. but it's useful for basic stuff.

optimization -- some different types of portfolio optimization. simple MVO and Black-Litterman (both single-period) with derivations and explanations.

options -- messing around with Black-Scholes. derivation and basic pricing. not my wheelhouse but good applied math / programming / pricing.
