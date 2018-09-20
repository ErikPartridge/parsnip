use num::cast::ToPrimitive;
use std::collections::HashMap;
extern crate num;
use num::Unsigned;

/// Compute the gini impurity of a dataset. 
/// 
/// Returns a float, 0 representing a perfectly pure dataset. Normal distribution: ~0.33
/// 
/// By default, any empty dataset will return a gini of 1.0. This may be unexpected behaviour.
/// ```
/// use parsnip::gini;
/// assert_eq!(gini(&vec![0_usize, 0, 0, 1]), 0.375);
/// ```
pub fn gini<T: Unsigned + ToPrimitive>(data: &[T]) -> f32 {
    if data.len() == 0 {
        return 1.0;
    } 
    fn p_squared(count: usize, len: f32) -> f32 {
        let p = count as f32 / len;
        p * p
    }
    let len = data.len() as f32;
    let mut count = HashMap::new();
    for ref value in data {
        *count.entry(value.to_u128().unwrap()).or_insert(0) += 1;
    }
    let counts: Vec<u128> = count.into_iter().map(|(_, c)| c).collect();
    let indiv : Vec<f32> = counts.iter().map(|x| p_squared(*x, len)).collect();
    let sum : f32 = indiv.iter().sum();
    1.0 - sum
}

/// The categorical accuracy of a dataset
/// 
/// Returns a float where 1.0 is a perfectly accurate dataset
/// ```
/// use parsnip::categorical_accuracy;
/// let pred : Vec<u16> = vec![0, 0, 0 , 1, 2];
/// let actual : Vec<u16> = vec![1, 1, 1, 1, 2];
/// assert_eq!(categorical_accuracy(&pred, &actual), 0.4);
/// ```
pub fn categorical_accuracy<T: Unsigned + ToPrimitive>(pred: &[T], actual: &[T]) -> f32 {
    assert_eq!(pred.len(), actual.len());
    let truthy =  pred.iter().zip(actual).filter(|(x,y)| x == y).count();
    return truthy as f32 / pred.len() as f32;
}

fn class_precision<T: Unsigned + ToPrimitive>(pred: &[T], actual: &[T], class: T) -> f32 {
    assert_eq!(pred.len(), actual.len());
    //First, get the map of all true positives
    let true_positives = pred.iter().zip(actual).filter(|(p, a)| p == a && **p == class).count() as f32;
    let all_positives = pred.iter().filter(|p| **p == class).count() as f32;
    if all_positives == 0.0 {
        0.0
    } else {
        true_positives / all_positives
    }
}

fn weighted_precision<T: Unsigned + ToPrimitive + Ord + Clone>(pred: &[T], actual: &[T]) -> f32 {
    assert_eq!(pred.len(), actual.len());
    let mut classes : Vec<&T> = pred.into_iter().collect();
    let mut class_weights = HashMap::new();
    classes.sort();
    classes.dedup();
    for value in classes.clone() {
        class_weights.insert(value.to_u128().unwrap(), actual.iter().filter(|a| *a == value).count() as f32 / actual.len() as f32);
    }
    return classes.iter().map(|c| class_precision(pred, actual, (**c).clone()) * class_weights.get(&c.to_u128().unwrap()).unwrap()).sum();
}

fn macro_precision<T: Unsigned + ToPrimitive + Ord + Clone>(pred: &[T], actual: &[T]) -> f32 {
    assert_eq!(pred.len(), actual.len());
    let mut classes : Vec<&T> = pred.into_iter().collect();
    let mut class_weights = HashMap::new();
    classes.sort();
    classes.dedup();
    for value in classes.clone() {
        class_weights.insert(value.to_u128().unwrap(), 1.0 / actual.len() as f32);
    }
    return classes.iter().map(|c| class_precision(pred, actual, (**c).clone()) / classes.len() as f32).sum();
}

/// The precision of a dataset
/// 
/// Returns a float where a 1.0 is a perfectly precise result set
/// 
/// Supports macro and weighted averages
/// ```
/// use parsnip::precision;
/// 
/// let actual = vec![0_u8, 1, 2, 0, 1, 2];
/// let pred = vec![0_u8, 2, 1, 0, 0, 1];
/// 
/// assert_eq!(precision(&pred, &actual, Some("macro".to_string())), 0.22222222);
/// ```
pub fn precision<T: Unsigned + ToPrimitive + Ord + Clone>(pred: &[T], actual: &[T], average: Option<String>) -> f32 {
    match average {
        None => macro_precision(pred, actual),
        Some(string) => match string.as_ref() {
            "macro" => macro_precision(pred, actual),
            "weighted" => weighted_precision(pred, actual),
            _ => panic!("invalid averaging type")
        }
    }
}

fn class_recall<T: Unsigned>(pred: &[T], actual: &[T], class: T) -> f32 {
    assert_eq!(pred.len(), actual.len());
    let true_positives = pred.iter().zip(actual).filter(|(p, a)| p == a && **a == class).count() as f32;
    let tp_fn = actual.iter().filter(|a| **a == class).count() as f32;
    if tp_fn == 0.0 {
        0.0
    } else {
        true_positives / tp_fn
    }
}

fn weighted_recall<T: Unsigned + ToPrimitive + Ord + Clone>(pred: &[T], actual: &[T]) -> f32 {
    assert_eq!(pred.len(), actual.len());
    let mut classes : Vec<&T> = pred.into_iter().collect();
    let mut class_weights = HashMap::new();
    classes.sort();
    classes.dedup();
    for value in classes.clone() {
        class_weights.insert(value.to_u128().unwrap(), actual.iter().filter(|a| **a == *value).count() as f32 / actual.len() as f32);
    }
    return classes.iter().map(|c| class_recall(pred, actual, (*c).clone()) * class_weights.get(&c.to_u128().unwrap()).unwrap()).sum();
}

fn macro_recall<T: Unsigned + ToPrimitive + Ord + Clone>(pred: &[T], actual: &[T]) -> f32 {
    assert_eq!(pred.len(), actual.len());
    let mut classes : Vec<&T> = pred.into_iter().collect();
    let mut class_weights = HashMap::new();
    classes.sort();
    classes.dedup();
    for value in classes.clone() {
        class_weights.insert(value.to_u128().unwrap(), 1.0 / actual.len() as f32);
    }
    return classes.iter().map(|c| class_recall(pred, actual, (*c).clone()) / classes.len() as f32).sum();
}

/// The recall of a dataset
/// 
/// Returns a float where a 1.0 is a perfectly recalled result set
/// 
/// Supports macro and weighted averages
/// ```
/// use parsnip::recall;
/// 
/// let actual = vec![0_u8, 1, 2, 0, 1, 2];
/// let pred = vec![0_u8, 2, 1, 0, 0, 1];
/// 
/// assert_eq!(recall(&pred, &actual, Some("macro".to_string())), 0.333333334);
/// ```
pub fn recall<T: Unsigned + ToPrimitive + Ord + Clone>(pred: &[T], actual: &[T], average: Option<String>) -> f32 {
    match average {
        None => macro_recall(pred, actual),
        Some(string) => match string.as_ref() {
            "macro" => macro_recall(pred, actual),
            "weighted" => weighted_recall(pred, actual),
            _ => panic!("invalid averaging type")
        }
    }
}

fn macro_f1<T: Unsigned + ToPrimitive + Ord + Clone>(pred: &[T], actual: &[T]) -> f32 {
    let recall = macro_recall(pred, actual);
    let precision = macro_precision(pred, actual);
    2.0 * (recall * precision) / (recall + precision)
}

fn weighted_f1<T: Unsigned + ToPrimitive + Ord + Clone>(pred: &[T], actual: &[T]) -> f32 {
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
/// use parsnip::f1_score;
/// 
/// let actual = vec![0_u8, 1, 2, 0, 1, 2];
/// let pred = vec![0_u8, 2, 1, 0, 0, 1];
/// 
/// assert_eq!(f1_score(&pred, &actual, Some("macro".to_string())), 0.26666665);
/// assert_eq!(f1_score(&pred, &actual, Some("weighted".to_string())), 0.26666668);
/// ```
pub fn f1_score<T: Unsigned + ToPrimitive + Ord + Clone>(pred: &[T], actual: &[T], average: Option<String>) -> f32 {
    match average {
        None => macro_f1(pred, actual),
        Some(string) => match string.as_ref() {
            "macro" => macro_f1(pred, actual),
            "weighted" => weighted_f1(pred, actual),
            _ => panic!("invalid averaging type")
        }
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
/// let actual = vec![0_u8, 1, 2, 0, 0];
/// let pred = vec![0_u8, 2, 1, 0, 1];
/// 
/// assert_eq!(hamming_loss(&pred, &actual), 0.6);
/// ```
pub fn hamming_loss<T: Unsigned + ToPrimitive>(pred: &[T], actual: &[T]) -> f32 {
    return 1.0 - categorical_accuracy(pred, actual);
}

fn macro_fbeta_score<T: Unsigned + ToPrimitive + Ord + Clone>(pred: &[T], actual: &[T], beta: f32) -> f32 {
    let precision = macro_precision(pred, actual);
    let recall = macro_recall(pred, actual);
    let top = (1.0 + beta * beta)  * (recall * precision);
    let bottom = (beta * beta * precision) + recall;
    top / bottom
}

fn weighted_fbeta_score<T: Unsigned + ToPrimitive + Ord + Clone>(pred: &[T], actual: &[T], beta: f32) -> f32 {
    let precision = weighted_precision(pred, actual);
    let recall = weighted_recall(pred, actual);
    let top = (1.0 + beta * beta)  * (recall * precision);
    let bottom = (beta * beta * precision) + recall;
    top / bottom
}

/// The fbeta of a dataset
/// 
/// Returns the fbeta score [0, 1]
/// 
/// Supports macro and weighted averages
/// ```
/// use parsnip::fbeta_score;
/// 
/// let actual = vec![0_u8, 1, 2, 0, 1, 2];
/// let pred = vec![0_u8, 2, 1, 0, 0, 1];
/// 
/// assert_eq!(fbeta_score(&pred, &actual, 0.5, Some("macro".to_string())), 0.23809524);
/// assert_eq!(fbeta_score(&pred, &actual, 0.5, Some("weighted".to_string())), 0.23809527);
/// ```
pub fn fbeta_score<T: Unsigned + ToPrimitive + Ord + Clone>(pred: &[T], actual: &[T], beta: f32, average: Option<String>) -> f32 {
    match average {
        None => macro_fbeta_score(pred, actual, beta),
        Some(string) => match string.as_ref() {
            "macro" => macro_fbeta_score(pred, actual, beta),
            "weighted" => weighted_fbeta_score(pred, actual, beta),
            _ => panic!("invalid averaging type")
        }
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
/// let actual = vec![0_u8, 2, 1, 3];
/// let pred = vec![0_u8, 1, 2, 3];
/// 
/// assert_eq!(jaccard_similiarity_score(&pred, &actual), 0.5);
/// ```
pub fn jaccard_similiarity_score<T: Unsigned + ToPrimitive>(pred: &[T], actual: &[T]) -> f32 {
    return categorical_accuracy(pred, actual);
}



#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_gini() {
        let vec : Vec<usize> = vec![0, 0, 0, 1];
        assert_eq!(0.375, gini(&vec));
        let v2 : Vec<usize> = vec![0, 0];
        assert_eq!(0.0, gini(&v2));
        let mut v3: Vec<usize> = vec![0];
        v3.pop();
        assert_eq!(1.0, gini(&v3));
    }

    #[test]
    fn test_categorical_accuracy() {
        let pred : Vec<u16> = vec![0, 1, 0, 1, 0, 1];
        let real : Vec<u16> = vec![0, 0, 0, 0, 1, 0];
        assert_eq!(0.33333334, categorical_accuracy(&pred, &real));
    }

    #[test]
    fn test_class_precision() {
        let actual = vec![0_u16, 1, 2, 0, 1, 2];
        let pred : Vec<u16> = vec![0, 2, 1, 0, 0, 1];
        assert_eq!(0.6666667, class_precision(&pred, &actual, 0));
    }

    #[test]
    fn test_class_recall() {
        let actual = vec![0_u16, 1, 2, 0, 0, 0];
        let pred = vec![0_u16, 2, 1, 0, 0, 1];
        assert_eq!(0.75, class_recall(&pred, &actual, 0));
    }

    #[test]
    fn test_weighted_precision() {
        let actual = vec![0_u16, 1, 2, 0, 1, 2];
        let pred = vec![0_u16, 2, 1, 0, 0, 1];
        assert_eq!(0.22222224, weighted_precision(&pred, &actual));
    }

    #[test]
    fn test_macro_precision() {
        let actual = vec![0_u16, 1, 2, 0, 1, 2];
        let pred = vec![0_u16, 2, 1, 0, 0, 1];
        assert_eq!(0.22222222, macro_precision(&pred, &actual));
    }

        #[test]
    fn test_macro_recall() {
        let actual = vec![0_u8, 1, 2, 0, 1, 2];
        let pred = vec![0_u8, 2, 1, 0, 0, 1];
        assert_eq!(0.33333334, macro_recall(&pred, &actual));
    }

        #[test]
    fn test_weighted_recall() {
        let actual = vec![0_u8, 1, 2, 0, 1, 2];
        let pred = vec![0_u8, 2, 1, 0, 0, 1];
        assert_eq!(0.333333334, weighted_recall(&pred, &actual));
    }

    #[test]
    fn test_f1_score() {
        let actual = vec![0_u8, 1, 2, 0, 1, 2];
        let pred = vec![0_u8, 2, 1, 0, 0, 1];
        assert_eq!(f1_score(&pred, &actual, Some("macro".to_string())), 0.26666665);
    }
}
