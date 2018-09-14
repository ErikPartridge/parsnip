# parsnip

High performance data science metrics implemented in Rust.

### Goals

Parsnip aims to supply a wide variety of useful metrics for data science. It presently has, and should always have zero dependencies, aims to not unneccessarily slow things down, and should guarantee accuracy. Parsnip should scale well to handle anything from 1 to 10M data points, and hopefully be thread safe. There should be no unsafe code in Parsnip.


### Contributing

Pull requests are always welcome! I ask that you keep with the convention of the package `fn something(pred: &[type], actual: &[type], ...)`. Performance PRs are always tremendously appreciated.

### Why the name parsnip?

I'm not sure. I wanted something natural, and it felt right. Plus, it's relatively short to type and distinct from other packages on [crates.io](https://crates.io). 

### Roadmap

I'm first aiming to finish broad support for categorical data—I want it to cover most, if not all of the SciKit Learn algorithms—at a minimum the non-highly complex ones (AUC might be a bit later). Plus, I'll add in some other useful once such as gini impurity.


### Change log

##### 0.1.3
Added f1_score support

Documenation is available at [docs.rs](https://docs.rs/parsnip#)