In the provided notebook `lstm`, all three tasks are included: **monovariate**, **multivariate**, and **future temperature**.  
The prepared dataset files (`.pkl`) are:

- `X_train_multi`, `X_test_multi`  
- `y_train_multi`, `y_test_multi`

These files are publicly available in the following Google Drive folder:  
🔗 https://drive.google.com/drive/folders/1vVHI4weXcis2ZBlENm0u1hxLPbppkkBa?usp=sharing  
(This folder also contains saved checkpoints and experiment results.)

> **Note:** "multi" stands for *multidimensional*, meaning that both `O₃` and `temperature` are already stored together.

In the experiments, I plotted several sample predictions to identify potential differences across the tasks. However, no significant differences were visible.  
In the final notebook, you will see samples with indices **0**, **5**, and **10**.

In the separate notebook `7_multi_model_plotting`, I also plotted test samples with indices **0**, **50**, and **100**.

Based on the provided Drive folder, reproducing the experiments should not be a problem.  
If any issues arise, feel free to reach out.

---

As you can see, it's basically the same code with small adjustments according to Task 1, 2, or 3. I always load my prepared data from the first two notebooks (I will make this folder public and will upload it to GitHub as well).

The second code block just differs in two points from the first one: we extract the first two columns from the prepared dataset since we want to do a multivariate prediction. (Y stays O₃ only.)

In the third task, I made the assumption (as mentioned in the code comments) that we should only use the future temperature.  
**In the context of the LSTM notebook**, if this assumption is incorrect, the following two lines:

```python
X_combined_train[:, context_window:, 1] = y_temp_train  
X_combined_test[:, context_window:, 1] = y_temp_test

# to this

X_combined_train[:, :context_window, 1] = X_temp_train  
X_combined_test[:, :context_window, 1] = X_temp_test

X_combined_train[:, context_window:, 1] = y_temp_train  
X_combined_test[:, context_window:, 1] = y_temp_test
