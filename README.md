# parsnip

[![Coverage Status](https://coveralls.io/repos/github/ErikPartridge/parsnip/badge.svg?branch=master)](https://coveralls.io/github/ErikPartridge/parsnip?branch=master) [![Docs](https://docs.rs/parsnip/badge.svg)](https://docs.rs/parsnips)

Data science metrics for Rust (categorical at the moment, but under active development).

### Goals

Parsnip aims to supply a wide variety of useful metrics for data science. Generally, I draw inspiration from [SciKit Learn's metrics](http://scikit-learn.org/stable/modules/classes.html) in terms of what to include. However, I'll add other features as contributors suggest or as I come across them (for example, Gini Impurity).

Ultimately, Parsnip should support a wide variety of linear algebra packages, including some with GPU support. Wherever possible, I will also endeavour to make Parsnip as performant as possible to avoid slowing down training.


### Contributing

Pull requests are always welcome! I ask that you keep with the convention of the package `fn something(pred: &[type], actual: &[type], ...)`. Performance PRs are always tremendously appreciated.

### Why the name parsnip?

I'm not sure. I wanted something natural, and it felt right. Plus, it's relatively short to type and distinct from other packages on [crates.io](https://crates.io). 

### Roadmap

I'm first aiming to finish broad support for categorical data. Code quality improvements, and greater unit test coverage would also be preferred. Before getting to 1.0, support for types other than slices is desirable.

### Change log

#### 0.3.0
Support for generic types, better error handling and documentation. Substantial breaking changes in this version.

#### 0.2.2
Mostly bug fixes, a few additions

#### 0.2.0
Added numerous different functions for categorical accuracy, bumping to 0.2.0 as a result. I consider most of these now stable.

#### 0.1.3
Added f1_score support

Documenation is available at [docs.rs](https://docs.rs/parsnip#)