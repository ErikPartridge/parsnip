use std::collections::HashMap;
use std::collections::HashSet;
use std::hash::Hash;

/// Compute the gini impurity of a dataset.
///
/// Returns a float, 0 representing a perfectly pure dataset. Normal distribution: ~0.33
///
/// By default, any empty dataset will return a gini of 1.0. This may be unexpected behaviour.
/// ```
/// use parsnip::gini;
/// assert_eq!(gini(&vec![0, 0, 0, 1]), 0.375);
/// ```
pub fn gini<T>(data: &[T]) -> f32
where
    T: Eq,
    T: Hash,
{
    if data.is_empty() {
        return 1.0;
    }
    fn p_squared(count: usize, len: f32) -> f32 {
        let p = count as f32 / len;
        p * p
    }
    let len = data.len() as f32;
    let mut count = HashMap::new();
    for value in data {
        *count.entry(value).or_insert(0) += 1;
    }
    let sum: f32 = count
        .into_iter()
        .map(|(_, c)| c)
        .map(|x| p_squared(x, len))
        .sum();
    1.0 - sum
}

/// The categorical accuracy of a dataset
///
/// Returns a float where 1.0 is a perfectly accurate dataset
/// ```
/// use parsnip::categorical_accuracy;
/// let pred = vec![0, 0, 0 , 1, 2];
/// let actual = vec![1, 1, 1, 1, 2];
/// assert_eq!(categorical_accuracy(&pred, &actual), 0.4);
/// ```
pub fn categorical_accuracy<T>(pred: &[T], actual: &[T]) -> f32
where
    T: Eq,
{
    assert_eq!(pred.len(), actual.len());
    let truthy = pred.iter().zip(actual).filter(|(x, y)| x == y).count();
    truthy as f32 / pred.len() as f32
}

fn class_precision<T>(pred: &[T], actual: &[T], class: &T) -> f32
where
    T: Eq,
{
    assert_eq!(pred.len(), actual.len());
    //First, get the map of all true positives
    let true_positives = pred
        .iter()
        .zip(actual)
        .filter(|(p, a)| p == a && **p == *class)
        .count() as f32;
    let all_positives = pred.iter().filter(|p| **p == *class).count() as f32;
    if all_positives == 0.0 {
        0.0
    } else {
        true_positives / all_positives
    }
}

fn weighted_precision<T>(pred: &[T], actual: &[T]) -> f32
where
    T: Eq,
    T: Hash,
{
    assert_eq!(pred.len(), actual.len());
    let classes: HashSet<_> = pred.into_iter().collect();
    let mut class_weights = HashMap::new();
    for value in &classes {
        class_weights.insert(
            value,
            actual.iter().filter(|a| *a == *value).count() as f32 / actual.len() as f32,
        );
    }
    classes
        .iter()
        .map(|c| class_precision(pred, actual, &c) * class_weights[c])
        .sum()
}

fn macro_precision<T>(pred: &[T], actual: &[T]) -> f32
where
    T: Eq,
    T: Hash,
{
    assert_eq!(pred.len(), actual.len());
    let classes: HashSet<_> = pred.into_iter().collect();
    let mut class_weights = HashMap::new();
    for value in classes.clone() {
        class_weights.insert(value, 1.0 / actual.len() as f32);
    }
    classes
        .iter()
        .map(|c| class_precision(pred, actual, c) / classes.len() as f32)
        .sum()
}

/// The type of score averaging strategy employed in the calculation of
/// precision, recall, or F-measure.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Average {
    /// Macro averaging (averaged across classes or labels).
    Macro,
    /// Averaging across classes, weighted by the number of true instances.
    Weighted,
}

impl Default for Average {
    /// The default average strategy is `Average::Macro`.
    fn default() -> Self {
        Average::Macro
    }
}

/// The precision of a dataset
///
/// Returns a float where a 1.0 is a perfectly precise result set
///
/// Supports macro and weighted averages
/// ```
/// # extern crate parsnip;
/// #[macro_use] extern crate approx; // for approximate equality check
/// use parsnip::{Average, precision};
///
/// let actual = vec![0, 1, 2, 0, 1, 2];
/// let pred = vec![0, 2, 1, 0, 0, 1];
/// 
/// assert_ulps_eq!(precision(&pred, &actual, Average::Macro), 0.22222222);
/// ```
pub fn precision<T>(pred: &[T], actual: &[T], average: Average) -> f32
where
    T: Eq,
    T: Hash,
{
    match average {
        Average::Macro => macro_precision(pred, actual),
        Average::Weighted => weighted_precision(pred, actual),
    }
}

fn class_recall<T>(pred: &[T], actual: &[T], class: &T) -> f32
where
    T: Eq,
{
    assert_eq!(pred.len(), actual.len());
    let true_positives = pred
        .iter()
        .zip(actual)
        .filter(|(p, a)| p == a && **a == *class)
        .count() as f32;
    let tp_fn = actual.iter().filter(|a| **a == *class).count() as f32;
    if tp_fn == 0.0 {
        0.0
    } else {
        true_positives / tp_fn
    }
}

fn weighted_recall<T>(pred: &[T], actual: &[T]) -> f32
where
    T: Eq,
    T: Hash,
{
    assert_eq!(pred.len(), actual.len());
    let classes: HashSet<_> = pred.into_iter().collect();
    let mut class_weights = HashMap::new();
    for value in &classes {
        class_weights.insert(
            value,
            actual.iter().filter(|a| **a == **value).count() as f32 / actual.len() as f32,
        );
    }
    classes
        .iter()
        .map(|c| class_recall(pred, actual, &c) * class_weights[c])
        .sum()
}

fn macro_recall<T>(pred: &[T], actual: &[T]) -> f32
where
    T: Eq,
    T: Hash,
{
    assert_eq!(pred.len(), actual.len());
    let classes: HashSet<_> = pred.into_iter().collect();
    classes
        .iter()
        .map(|c| class_recall(pred, actual, *c) / classes.len() as f32)
        .sum()
}

/// The recall of a dataset
///
/// Returns a float where a 1.0 is a perfectly recalled result set
///
/// Supports macro and weighted averages
/// ```
/// # extern crate parsnip;
/// #[macro_use] extern crate approx; // for approximate equality check
/// use parsnip::{Average, recall};
///
/// let actual = vec![0, 1, 2, 0, 1, 2];
/// let pred = vec![0, 2, 1, 0, 0, 1];
/// 
/// assert_ulps_eq!(recall(&pred, &actual, Average::Macro), 0.333333333);
/// ```
pub fn recall<T>(pred: &[T], actual: &[T], average: Average) -> f32
where
    T: Eq,
    T: Hash,
{
    match average {
        Average::Macro => macro_recall(pred, actual),
        Average::Weighted => weighted_recall(pred, actual),
    }
}

fn macro_f1<T>(pred: &[T], actual: &[T]) -> f32
where
    T: Eq,
    T: Hash,
{
    let recall = macro_recall(pred, actual);
    let precision = macro_precision(pred, actual);
    2.0 * (recall * precision) / (recall + precision)
}

fn weighted_f1<T>(pred: &[T], actual: &[T]) -> f32
where
    T: Eq,
    T: Hash,
{
    let recall = weighted_recall(pred, actual);
    let precision = weighted_precision(pred, actual);
    2.0 * (recall * precision) / (recall + precision)
}

/// The f1 score of a dataset
///
/// Returns an f1 score where 1 is perfect and 0 is atrocious.
///
/// Supports macro and weighted averages
/// ```
/// # extern crate parsnip;
/// #[macro_use] extern crate approx; // for approximate equality check
/// use parsnip::{Average, f1_score};
///
/// let actual = vec![0, 1, 2, 0, 1, 2];
/// let pred = vec![0, 2, 1, 0, 0, 1];
/// 
/// assert_ulps_eq!(f1_score(&pred, &actual, Average::Macro), 0.26666666);
/// assert_ulps_eq!(f1_score(&pred, &actual, Average::Weighted), 0.26666666);
/// ```
pub fn f1_score<T>(pred: &[T], actual: &[T], average: Average) -> f32
where
    T: Eq,
    T: Hash,
{
    match average {
        Average::Macro => macro_f1(pred, actual),
        Average::Weighted => weighted_f1(pred, actual),
    }
}

/// The hamming loss of a dataset
///
/// Returns the hamming loss which is the percentage of items which are misclassified [0, 1]
///
/// Supports macro and weighted averages
/// ```
/// use parsnip::hamming_loss;
///
/// let actual = vec![0, 1, 2, 0, 0];
/// let pred = vec![0, 2, 1, 0, 1];
///
/// assert_eq!(hamming_loss(&pred, &actual), 0.6);
/// ```
pub fn hamming_loss<T>(pred: &[T], actual: &[T]) -> f32
where
    T: Eq,
{
    1.0 - categorical_accuracy(pred, actual)
}

fn macro_fbeta_score<T>(pred: &[T], actual: &[T], beta: f32) -> f32
where
    T: Eq,
    T: Hash,
{
    let precision = macro_precision(pred, actual);
    let recall = macro_recall(pred, actual);
    let top = (1.0 + beta * beta) * (recall * precision);
    let bottom = (beta * beta * precision) + recall;
    top / bottom
}

fn weighted_fbeta_score<T>(pred: &[T], actual: &[T], beta: f32) -> f32
where
    T: Eq,
    T: Hash,
{
    let precision = weighted_precision(pred, actual);
    let recall = weighted_recall(pred, actual);
    let top = (1.0 + beta * beta) * (recall * precision);
    let bottom = (beta * beta * precision) + recall;
    top / bottom
}

/// The fbeta of a dataset
///
/// Returns the fbeta score [0, 1]
///
/// Supports macro and weighted averages
/// ```
/// # extern crate parsnip;
/// #[macro_use] extern crate approx; // for approximate equality check
/// use parsnip::{Average, fbeta_score};
///
/// let actual = vec![0, 1, 2, 0, 1, 2];
/// let pred = vec![0, 2, 1, 0, 0, 1];
/// 
/// assert_ulps_eq!(fbeta_score(&pred, &actual, 0.5, Average::Macro), 0.23809524);
/// assert_ulps_eq!(fbeta_score(&pred, &actual, 0.5, Average::Weighted), 0.23809527);
/// ```
pub fn fbeta_score<T>(pred: &[T], actual: &[T], beta: f32, average: Average) -> f32
where
    T: Eq,
    T: Hash,
{
    match average {
        Average::Macro => macro_fbeta_score(pred, actual, beta),
        Average::Weighted => weighted_fbeta_score(pred, actual, beta),
    }
}

/// The jaccard similarity of a dataset
///
/// Returns the jaccard similarity score which for our purposes is effectively categorical accuracy [0, 1]
///
/// Supports macro and weighted averages
/// ```
/// use parsnip::jaccard_similiarity_score;
///
/// let actual = vec![0, 2, 1, 3];
/// let pred = vec![0, 1, 2, 3];
///
/// assert_eq!(jaccard_similiarity_score(&pred, &actual), 0.5);
/// ```
pub fn jaccard_similiarity_score<T>(pred: &[T], actual: &[T]) -> f32
where
    T: Eq,
{
    categorical_accuracy(pred, actual)
}

#[cfg(test)]
#[macro_use] extern crate approx;

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_gini() {
        let vec = vec![0, 0, 0, 1];
        assert_ulps_eq!(0.375, gini(&vec));
        let v2 = vec![0, 0];
        assert_ulps_eq!(0.0, gini(&v2));
        let mut v3 = vec![0];
        v3.pop();
        assert_ulps_eq!(1.0, gini(&v3));
    }

    #[test]
    fn test_categorical_accuracy() {
        let pred = vec![0, 1, 0, 1, 0, 1];
        let real = vec![0, 0, 0, 0, 1, 0];
        assert_ulps_eq!(0.33333333, categorical_accuracy(&pred, &real));
    }

    #[test]
    fn test_class_precision() {
        let actual = vec![0, 1, 2, 0, 1, 2];
        let pred = vec![0, 2, 1, 0, 0, 1];
        assert_ulps_eq!(0.6666666, class_precision(&pred, &actual, &0));
    }

    #[test]
    fn test_class_recall() {
        let actual = vec![0, 1, 2, 0, 0, 0];
        let pred = vec![0, 2, 1, 0, 0, 1];
        assert_ulps_eq!(0.75, class_recall(&pred, &actual, &0));
    }

    #[test]
    fn test_weighted_precision() {
        let actual = vec![0, 1, 2, 0, 1, 2];
        let pred = vec![0, 2, 1, 0, 0, 1];
        assert_ulps_eq!(0.22222222, weighted_precision(&pred, &actual));
    }

    #[test]
    fn test_macro_precision() {
        let actual = vec![0, 1, 2, 0, 1, 2];
        let pred = vec![0, 2, 1, 0, 0, 1];
        assert_ulps_eq!(0.22222222, macro_precision(&pred, &actual));
    }

    #[test]
    fn test_macro_recall() {
        let actual = vec![0, 1, 2, 0, 1, 2];
        let pred = vec![0, 2, 1, 0, 0, 1];
        assert_ulps_eq!(0.33333333, macro_recall(&pred, &actual));
    }

    #[test]
    fn test_weighted_recall() {
        let actual = vec![0, 1, 2, 0, 1, 2];
        let pred = vec![0, 2, 1, 0, 0, 1];
        assert_ulps_eq!(0.333333333, weighted_recall(&pred, &actual));
    }

    #[test]
    fn test_f1_score() {
        let actual = vec![0, 1, 2, 0, 1, 2];
        let pred = vec![0, 2, 1, 0, 0, 1];
        assert_ulps_eq!(f1_score(&pred, &actual, Average::Macro), 0.26666665);
    }
}
