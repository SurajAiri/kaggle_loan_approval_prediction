# feature transformation techniques

## based on Skewness & Kurtosis

| **Skewness** | **Kurtosis**  | **Distribution Shape**             | **Recommended Transformation**                                   | **Example Code**                                                                                                              |
| ------------ | ------------- | ---------------------------------- | ---------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------- |
| Skewness ≈ 0 | Kurtosis ≈ 3  | Symmetrical, normal                | No transformation needed                                         | -                                                                                                                             |
| Skewness > 0 | Kurtosis > 3  | Right-skewed, heavy tails          | **Log** (if large values) or **Square root** (for moderate skew) | `X['log_feature'] = np.log1p(X['feature'])` <br> or <br> `X['sqrt_feature'] = np.sqrt(X['feature'])`                          |
| Skewness > 1 | Kurtosis > 3  | Strong right-skew                  | **Log transformation** or **Box-Cox** (if no zeros)              | `X['log_feature'] = np.log1p(X['feature'])`                                                                                   |
| Skewness > 2 | Kurtosis >> 3 | Highly right-skewed, many outliers | **Log transformation**, **Capping Outliers**                     | `X['log_feature'] = np.log1p(X['feature'])` <br> or Winsorization for capping                                                 |
| Skewness < 0 | Kurtosis > 3  | Left-skewed, heavy tails           | **Reflect and log/square root**                                  | `X['log_feature'] = np.log1p(X['max_val'] - X['feature'])`                                                                    |
| Skewness ≈ 0 | Kurtosis > 3  | Symmetrical, heavy tails           | **Box-Cox** or **Yeo-Johnson**                                   | `from sklearn.preprocessing import PowerTransformer` <br> `X['transformed'] = PowerTransformer().fit_transform(X['feature'])` |
| Skewness ≈ 0 | Kurtosis < 3  | Symmetrical, light tails           | **No transformation**                                            | -                                                                                                                             |
| Skewness > 0 | Kurtosis < 3  | Right-skewed, light tails          | **Square root transformation**                                   | `X['sqrt_feature'] = np.sqrt(X['feature'])`                                                                                   |
| Skewness < 0 | Kurtosis < 3  | Left-skewed, light tails           | **Reflect and square root**                                      | `X['sqrt_feature'] = np.sqrt(X['max_val'] - X['feature'])`                                                                    |

### General Guidelines:

- **Logarithmic transformation**: Best for **highly right-skewed data** with **large outliers**.
- **Square root transformation**: Effective for **moderately right-skewed** data with **lighter tails**.
- **Box-Cox** or **Yeo-Johnson**: Good for **non-normal data** (especially if negative values exist). These techniques are flexible and can adjust for both skewness and kurtosis.
- **Winsorization or capping outliers**: Useful when kurtosis is extremely high, indicating the presence of severe outliers.

This table provides a quick reference based on the skewness and kurtosis statistics to choose an appropriate transformation.

## Outlier handling

### **Summary of Techniques**

| Method                        | When to Use                                                        | Pros                                            | Cons                                                  |
| ----------------------------- | ------------------------------------------------------------------ | ----------------------------------------------- | ----------------------------------------------------- |
| Remove Outliers               | If outliers are errors or noise                                    | Simplifies data, improves model performance     | Risk of losing valuable rare information              |
| Winsorize (Capping)           | When outliers are extreme but still contain useful information     | Retains data while reducing extreme values      | May oversimplify or bias data                         |
| Log Transformation            | For highly skewed data, to compress extreme values                 | Reduces skewness and spreads data more normally | Can distort interpretation of the variable            |
| Box-Cox/Yeo-Johnson           | For non-normal data distributions, especially with positive values | Handles a wide range of distributions           | Can’t be applied to negative or zero values (Box-Cox) |
| Robust Scaling                | For datasets with outliers that shouldn’t be removed               | Less sensitive to extreme values                | Only applies to scaling, not removal                  |
| Imputation (Replace Outliers) | If you want to retain data but reduce outliers’ influence          | Retains dataset size, controls extreme impact   | Can introduce bias if not done carefully              |
| Outlier Flag                  | When you want to retain outliers but use them as a feature         | Provides additional feature to inform the model | Doesn’t directly mitigate impact of outliers          |

### Next Steps:

1. **Experiment with different methods** and track the impact on your model's performance.
2. **Use domain knowledge** to determine whether outliers are valuable or noise.
3. **Try multiple methods** on a validation set and compare model accuracy, precision, recall, or other relevant metrics to find what works best.
