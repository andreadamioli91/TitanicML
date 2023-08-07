class Utils:

    def parify_test_columns(self, train_transformed, test_transformed):
        for col in train_transformed.columns:
            if col not in test_transformed.columns:
                test_transformed[col] = 0