# QMCPy 2.2 Portfolio allocation to-do

## Demo

1. What are the differences between the two notebooks in `demos/portfolio`? Keep `portfolio_allocation_demo` and merge content from `archive.ipynb` into it if necessary. Afterwards, delete `archive.ipynb`. --DONE
2. Add Sobol and Halton samplers for weights generation in the demo --DONE
3. Create a single function to generate weights for all sampler types --DONE. See `gen_weights*()`.
4. Update visualizations in the demo with all QMC samplers and risk categories --DONE
5. Make a function for collecting data (?) --DONE (`evaluate_sampler_sharpe`)
6. Update start_date and end_date --DONE
7. Save price datasets to `data` folder --DONE
8. Save figures as .png in `images` --DONE
9. Use same colors for same samplers across all visualizations --DONE
10. Use white background for all visualizations --DONE
11. Implement a more optimal simplex transformation for low-discrepancy sequences --DONE (added `<base>_root` variants, using `SimplexTransform.root`, for all four base samplers; included in the Sharpe-ratio comparison, the Runtime comparison, the in-sample performance section, and both out-of-sample backtests.)
12. Clean up demo --DONE (numbered sections, extracted repeated plotting/computation into named helpers, consolidated imports, merged the redundant "Runtime comparison"/"Visualizing runtime" split into one section)
14. Improve documentation --DONE (see items 2 and 4 below)

## Documentation

1. Stop checking in .png or binary files into repository --DONE
2. Create `mkdocs.yml` documentation for the demo and make it like a blog --DONE (already listed in `mkdocs.yml`'s nav; notebook now opens with a title/author/motivation paragraph and reads section-by-section like a blog post, ending in a Conclusions section)
3. Add references to the demo--DONE
4. Add more text in Markdown cells before code cells to the demo --DONE (every section now has an explanatory intro; a few short setup/plumbing cells are left undocumented as self-explanatory)

## Tests

1. Utilize QMCPy's `replications` parameter for averaged results --DONE
2. Implement out-of-sample backtesting --DONE (see "Out-of-sample backtesting" sections, 10- and 40-ticker groups)
3. Make tb_portfolio_allocation_demo.py work in CI tests on Windows ---TODO, Brandon  
4. Implement rebalancing



## Poster

1. Update poster in Overleaf --TODO
