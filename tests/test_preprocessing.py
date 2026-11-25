import numpy as np

from melba.utils.preprocessing import BusinessFeatureMap


def test_quadratic_expansion_has_expected_count_and_column_order():
    standardized = np.array([[1.0, 2.0, 3.0], [-1.0, 0.0, 4.0]])
    feature_map = BusinessFeatureMap(quadratic_terms=True)

    expanded = feature_map._expand(standardized)

    expected = np.array(
        [
            [1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 4.0, 6.0, 9.0],
            [-1.0, 0.0, 4.0, 1.0, 0.0, -4.0, 0.0, 0.0, 16.0],
        ]
    )
    np.testing.assert_array_equal(expanded, expected)
    assert expanded.shape[1] == 3 + 3 * (3 + 1) // 2


def test_linear_and_quadratic_feature_counts():
    features = np.array([[0.0, 1.0], [2.0, 2.0], [4.0, 0.0], [6.0, 3.0]])

    linear = BusinessFeatureMap(quadratic_terms=False).fit_transform(features)
    quadratic = BusinessFeatureMap(quadratic_terms=True).fit_transform(features)

    assert linear.shape[1] == 2
    assert quadratic.shape[1] == 2 + 2 * (2 + 1) // 2


def test_transform_reuses_training_statistics_without_test_data_leakage():
    train = np.array([[0.0, 1.0], [2.0, 2.0], [4.0, 0.0], [6.0, 3.0]])
    test = np.array([[100.0, -20.0], [-50.0, 60.0]])
    feature_map = BusinessFeatureMap(quadratic_terms=True)

    mapped_train = feature_map.fit_transform(train)
    training_stats = tuple(
        values.copy()
        for values in (
            feature_map.input_mean_,
            feature_map.input_scale_,
            feature_map.term_mean_,
            feature_map.term_scale_,
        )
    )
    mapped_test = feature_map.transform(test)

    np.testing.assert_allclose(feature_map.transform(train), mapped_train)
    np.testing.assert_array_equal(feature_map.input_mean_, train.mean(axis=0))
    assert not np.array_equal(feature_map.input_mean_, np.vstack((train, test)).mean(axis=0))
    np.testing.assert_allclose(
        mapped_test,
        (feature_map._expand((test - feature_map.input_mean_) / feature_map.input_scale_)
         - feature_map.term_mean_)
        / feature_map.term_scale_,
    )
    for actual_stat, training_stat in zip(
        (
            feature_map.input_mean_,
            feature_map.input_scale_,
            feature_map.term_mean_,
            feature_map.term_scale_,
        ),
        training_stats,
    ):
        np.testing.assert_array_equal(actual_stat, training_stat)
