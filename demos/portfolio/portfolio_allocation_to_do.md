# QMCPy 2.2 Portfolio allocation to-do

0. What are the differences between the two notebooks in `demos/portfolio`? Keep `portfolio_allocation_demo` and merge content from `archive.ipynb` into it if necessary. Afterwards, delete `archive.ipynb`. --- TODO 
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
12. Use white background for all visualizations --- DONE
13. Stopping checking in .png or binary files into repository --- TODO
14. Clean up demo --- TODO
15. Add references to the demo --- TODO
16. Update poster in Overleaf --- TODO
