# QMCPy 2.2 Portfolio allocation to-do

0. What are the differences between the two notebooks in `demos/portfolio`?
1. Add Sobol and Halton samplers for weights generation in the demo --- DONE
2. Add more text in Markdown cells before code cells to the demo --- TODO
3. . Create a single function to generate weights for all sampler types --- DONE. See `gen_weights*()`.
4. Update visualizations in the demo with all QMC samplers and risk categories --- DONE
5. Implement a more optimal simplex transformation for low-discrepancy sequences --- TODO
6. Utilize QMCPy's `replications` parameter for averaged results --- TODO
7. Make a function for collecting data (?) --- DONE (`evaluate_sampler_sharpe`)
8. Update start_date and end_date --- DONE
9.  Save price datasets to `data` folder --- DONE
10. Save figures as .png in `images` --- DONE
11. Use same colors for same samplers across all visualizations --- DONE
12. Clean up demo --- TODO
13. Add references to the demo --- TODO
14. Update poster in Overleaf --- TODO
