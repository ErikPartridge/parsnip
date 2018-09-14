use std::collections::HashMap;

/// Compute the gini impurity of a dataset. 
/// 
/// Returns a float, 0 representing a perfectly pure dataset. Normal distribution: ~0.33
/// 
/// By default, any empty dataset will return a gini of 1.0. This may be unexpected behaviour.
/// ```
/// use parsnip::gini;
/// assert_eq!(gini(&vec![0, 0, 0, 1]), 0.375);
/// ```
pub fn gini(data: &[u64]) -> f32 {
    if data.len() == 0 {
        return 1.0;
    } 
    fn p_squared(count: usize, len: f32) -> f32 {
        let p = count as f32 / len;
        return p * p;
    }
    let len = data.len() as f32;
    let mut count = HashMap::new();
    for &value in data {
        *count.entry(value).or_insert(0) += 1;
    }
    let counts: Vec<usize> = count.into_iter().map(|(_, c)| c).collect();
    let indiv : Vec<f32> = counts.iter().map(|x| p_squared(*x, len)).collect();
    let sum : f32 = indiv.iter().sum();
    return 1.0 - sum;
}

/// The categorical accuracy of a dataset
/// Returns a float where 1.0 is a perfectly accurate dataset
/// ```
/// use parsnip::categorical_accuracy;
/// let pred = vec![0, 0, 0 , 1, 2];
/// let actual = vec![1, 1, 1, 1, 2];
/// assert_eq!(categorical_accuracy(&pred, &actual), 0.4);
/// ```
pub fn categorical_accuracy(pred: &[u64], actual: &[u64]) -> f32 {
    assert_eq!(pred.len(), actual.len());
    let bools =  pred.iter().zip(actual).map(|(x,y)| x == y);
    let truthy : Vec<bool> =  bools.filter(|b| *b).collect();
    return truthy.len() as f32 / pred.len() as f32;
}

fn class_precision(pred: &[u64], actual: &[u64], class: u64) -> f32 {
    assert_eq!(pred.len(), actual.len());
    let true_positives_map = pred.iter().zip(actual).map(|(p, a)| p == a && *p == class);
    let true_positives = true_positives_map.filter(|b| *b).count() as f32;
    let all_positives = pred.iter().map(|p| *p == class).filter(|b| *b).count() as f32;
    if all_positives == 0.0 {
        return 0.0;
    }
    return true_positives / all_positives;
}

fn weighted_precision(pred: &[u64], actual: &[u64]) -> f32 {
    assert_eq!(pred.len(), actual.len());
    let mut classes : Vec<u64> = pred.into_iter().map(|x| *x).collect();
    let mut class_weights = HashMap::new();
    classes.sort();
    classes.dedup();
    for value in classes.clone() {
        class_weights.insert(value, actual.iter().filter(|a| **a == value).count() as f32 / actual.len() as f32);
    }
    return classes.iter().map(|c| class_precision(pred, actual, *c) * class_weights.get(c).unwrap()).sum();
}

fn macro_precision(pred: &[u64], actual: &[u64]) -> f32 {
    assert_eq!(pred.len(), actual.len());
    let mut classes : Vec<u64> = pred.into_iter().map(|x| *x).collect();
    let mut class_weights = HashMap::new();
    classes.sort();
    classes.dedup();
    for value in classes.clone() {
        class_weights.insert(value, 1.0 / actual.len() as f32);
    }
    return classes.iter().map(|c| class_precision(pred, actual, *c) / classes.len() as f32).sum();
}

/// The precision of a dataset
/// Returns a float where a 1.0 is a perfectly precise result set
/// 
/// Supports macro and weighted averages
/// ```
/// use parsnip::precision;
/// 
/// let actual = vec![0, 1, 2, 0, 1, 2];
/// let pred = vec![0, 2, 1, 0, 0, 1];
/// 
/// assert_eq!(precision(&pred, &actual, Some("macro".to_string())), 0.22222222);
/// ```
pub fn precision(pred: &[u64], actual: &[u64], average: Option<String>) -> f32 {
    match average {
        None => return macro_precision(pred, actual),
        Some(string) => match string.as_ref() {
            "macro" => return macro_precision(pred, actual),
            "weighted" => return weighted_precision(pred, actual),
            _ => panic!("invalid averaging type")
        }
    }
}

fn class_recall(pred: &[u64], actual: &[u64], class: u64) -> f32 {
    assert_eq!(pred.len(), actual.len());
    let true_positives_map = pred.iter().zip(actual).map(|(p, a)| p == a && *a == class);
    let true_positives = true_positives_map.filter(|b| *b).count() as f32;
    let tp_fn = actual.iter().map(|a| *a == class).filter(|b| *b).count() as f32;
    if tp_fn == 0.0 {
        return 0.0;
    }
    return true_positives / tp_fn;
}

fn weighted_recall(pred: &[u64], actual: &[u64]) -> f32 {
    assert_eq!(pred.len(), actual.len());
    let mut classes : Vec<u64> = pred.into_iter().map(|x| *x).collect();
    let mut class_weights = HashMap::new();
    classes.sort();
    classes.dedup();
    for value in classes.clone() {
        class_weights.insert(value, actual.iter().filter(|a| **a == value).count() as f32 / actual.len() as f32);
    }
    return classes.iter().map(|c| class_recall(pred, actual, *c) * class_weights.get(c).unwrap()).sum();
}

fn macro_recall(pred: &[u64], actual: &[u64]) -> f32 {
    assert_eq!(pred.len(), actual.len());
    let mut classes : Vec<u64> = pred.into_iter().map(|x| *x).collect();
    let mut class_weights = HashMap::new();
    classes.sort();
    classes.dedup();
    for value in classes.clone() {
        class_weights.insert(value, 1.0 / actual.len() as f32);
    }
    return classes.iter().map(|c| class_recall(pred, actual, *c) / classes.len() as f32).sum();
}

/// The recall of a dataset
/// Returns a float where a 1.0 is a perfectly recalled result set
/// 
/// Supports macro and weighted averages
/// ```
/// use parsnip::recall;
/// 
/// let actual = vec![0, 1, 2, 0, 1, 2];
/// let pred = vec![0, 2, 1, 0, 0, 1];
/// 
/// assert_eq!(recall(&pred, &actual, Some("macro".to_string())), 0.333333334);
/// ```
pub fn recall(pred: &[u64], actual: &[u64], average: Option<String>) -> f32 {
    match average {
        None => return macro_recall(pred, actual),
        Some(string) => match string.as_ref() {
            "macro" => return macro_recall(pred, actual),
            "weighted" => return weighted_recall(pred, actual),
            _ => panic!("invalid averaging type")
        }
    }
}

fn macro_f1(pred: &[u64], actual: &[u64]) -> f32 {
    let recall = macro_recall(pred, actual);
    let precision = macro_precision(pred, actual);
    return 2.0 * (recall * precision) / (recall + precision);
}

fn weighted_f1(pred: &[u64], actual: &[u64]) -> f32 {
    let recall = weighted_recall(pred, actual);
    let precision = weighted_precision(pred, actual);
    return 2.0 * (recall * precision) / (recall + precision);

}

/// The recall of a dataset
/// Returns an f1 score where 1 is perfect and 0 is atrocious.
/// 
/// Supports macro and weighted averages
/// ```
/// use parsnip::f1_score;
/// 
/// let actual = vec![0, 1, 2, 0, 1, 2];
/// let pred = vec![0, 2, 1, 0, 0, 1];
/// 
/// assert_eq!(f1_score(&pred, &actual, Some("macro".to_string())), 0.26666665);
/// assert_eq!(f1_score(&pred, &actual, Some("weighted".to_string())), 0.26666668);
/// ```
pub fn f1_score(pred: &[u64], actual: &[u64], average: Option<String>) -> f32 {
    match average {
        None => return macro_f1(pred, actual),
        Some(string) => match string.as_ref() {
            "macro" => return macro_f1(pred, actual),
            "weighted" => return weighted_f1(pred, actual),
            _ => panic!("invalid averaging type")
        }
    }
}

/// The recall of a dataset
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
pub fn hamming_loss(pred: &[u64], actual: &[u64]) -> f32 {
    return 1.0 - categorical_accuracy(pred, actual);
}

fn macro_fbeta_score(pred: &[u64], actual: &[u64], beta: f32) -> f32 {
    let precision = macro_precision(pred, actual);
    let recall = macro_recall(pred, actual);
    let top = (1.0 + beta * beta)  * (recall * precision);
    let bottom = (beta * beta * precision) + recall;
    return top / bottom;
}

fn weighted_fbeta_score(pred: &[u64], actual: &[u64], beta: f32) -> f32 {
    let precision = weighted_precision(pred, actual);
    let recall = weighted_recall(pred, actual);
    let top = (1.0 + beta * beta)  * (recall * precision);
    let bottom = (beta * beta * precision) + recall;
    return top / bottom;
}

/// The recall of a dataset
/// Returns the hamming loss which is the percentage of items which are misclassified [0, 1]
/// 
/// Supports macro and weighted averages
/// ```
/// use parsnip::fbeta_score;
/// 
/// let actual = vec![0, 1, 2, 0, 1, 2];
/// let pred = vec![0, 2, 1, 0, 0, 1];
/// 
/// assert_eq!(fbeta_score(&pred, &actual, 0.5, Some("macro".to_string())), 0.23809524);
/// assert_eq!(fbeta_score(&pred, &actual, 0.5, Some("weighted".to_string())), 0.23809527);
/// ```
pub fn fbeta_score(pred: &[u64], actual: &[u64], beta: f32, average: Option<String>) -> f32 {
    match average {
        None => return macro_fbeta_score(pred, actual, beta),
        Some(string) => match string.as_ref() {
            "macro" => return macro_fbeta_score(pred, actual, beta),
            "weighted" => return weighted_fbeta_score(pred, actual, beta),
            _ => panic!("invalid averaging type")
        }
    }
}

/// The recall of a dataset
/// Returns the hamming loss which is the percentage of items which are misclassified [0, 1]
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
pub fn jaccard_similiarity_score(pred: &[u64], actual: &[u64]) -> f32 {
    return categorical_accuracy(pred, actual);
}


#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_gini() {
        let vec = vec![0, 0, 0, 1];
        assert_eq!(0.375, gini(&vec));
        let v2 = vec![0, 0];
        assert_eq!(0.0, gini(&v2));
        let mut v3 = vec![0];
        v3.pop();
        assert_eq!(1.0, gini(&v3));
    }

    #[test]
    fn test_categorical_accuracy() {
        let pred = vec![0, 1, 0, 1, 0, 1];
        let real = vec![0, 0, 0, 0, 1, 0];
        assert_eq!(0.33333334, categorical_accuracy(&pred, &real));
    }

    #[test]
    fn test_class_precision() {
        let actual = vec![0, 1, 2, 0, 1, 2];
        let pred = vec![0, 2, 1, 0, 0, 1];
        assert_eq!(0.6666667, class_precision(&pred, &actual, 0));
    }

    #[test]
    fn test_class_recall() {
        let actual = vec![0, 1, 2, 0, 0, 0];
        let pred = vec![0, 2, 1, 0, 0, 1];
        assert_eq!(0.75, class_recall(&pred, &actual, 0));
    }

    #[test]
    fn test_weighted_precision() {
        let actual = vec![0, 1, 2, 0, 1, 2];
        let pred = vec![0, 2, 1, 0, 0, 1];
        assert_eq!(0.22222224, weighted_precision(&pred, &actual));
    }

    #[test]
    fn test_macro_precision() {
        let actual = vec![0, 1, 2, 0, 1, 2];
        let pred = vec![0, 2, 1, 0, 0, 1];
        assert_eq!(0.22222222, macro_precision(&pred, &actual));
    }

        #[test]
    fn test_macro_recall() {
        let actual = vec![0, 1, 2, 0, 1, 2];
        let pred = vec![0, 2, 1, 0, 0, 1];
        assert_eq!(0.33333334, macro_recall(&pred, &actual));
    }

        #[test]
    fn test_weighted_recall() {
        let actual = vec![0, 1, 2, 0, 1, 2];
        let pred = vec![0, 2, 1, 0, 0, 1];
        assert_eq!(0.333333334, weighted_recall(&pred, &actual));
    }

    #[test]
    fn test_f1_score() {
        let actual = vec![0, 1, 2, 0, 1, 2];
        let pred = vec![0, 2, 1, 0, 0, 1];
        assert_eq!(f1_score(&pred, &actual, Some("macro".to_string())), 0.26666665);
    }
}
