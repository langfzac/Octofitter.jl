## Pair plot 

Diagonal entries show estimates of the marginal 
densities as well as the (0.16, 0.5, 0.84) 
quantiles (dotted lines). 
Off-diagonal entries show estimates of the pairwise 
densities. 

![](pair_plot.png)


## Trace plots 


![](trace_plot.png)


## Moments 

| **parameters** | **mean**  | **std**     | **mcse**    | **ess\_bulk** | **ess\_tail** | **rhat** | **ess\_per\_sec** |
|---------------:|----------:|------------:|------------:|--------------:|--------------:|---------:|------------------:|
| param\_1       | 3.08539   | 0.00128554  | 0.000161268 | 66.4899       | 68.825        | 1.02488  | missing           |
| param\_2       | 0.0924601 | 0.0625076   | 0.0161352   | 14.3295       | 28.088        | 1.03411  | missing           |
| param\_3       | -3.6831   | 0.102361    | 0.022282    | 12.0739       | 23.1182       | 1.07805  | missing           |
| param\_4       | -1.31015  | 0.418353    | 0.0242857   | 246.099       | 39.4792       | 1.00317  | missing           |
| param\_5       | 0.825051  | 0.0854907   | 0.0118769   | 31.0907       | 22.0573       | 1.02175  | missing           |
| param\_6       | -0.773139 | 1.12323     | 0.0658361   | 76.5235       | 21.5892       | 1.06189  | missing           |
| param\_7       | -1.08344  | 1.18835     | 0.0682159   | 149.375       | 27.4881       | 1.08427  | missing           |
| param\_8       | -1.93077  | 0.000794063 | 2.54277e-5  | 979.494       | 843.771       | 1.00132  | missing           |
 

```@raw html
<a href="Moments.csv">💾 CSV</a> 
```


## Cumulative traces 

For each iteration ``i``, shows the running average up to ``i``,
``\frac{1}{i} \sum_{n = 1}^{i} x_n``. 

![](cumulative_trace_plot.png)


## Local communication barrier 

When the global communication barrier is large, many chains may 
be required to obtain tempered restarts.

The local communication barrier can be used to visualize the cause 
of a high global communication barrier. For example, if there is a 
sharp peak close to a reference constructed from the prior, it may 
be useful to switch to a [variational approximation](https://pigeons.run/dev/variational/#variational-pt).

![](local_barrier.png)


## GCB estimation progress 

Estimate of the Global Communication Barrier (GCB) 
as a function of 
the adaptation round. 

The global communication barrier can be used 
to set the number of chains. 
The theoretical framework of [Syed et al., 2021](https://academic.oup.com/jrsssb/article/84/2/321/7056147)
yields that under simplifying assumptions, it is optimal to set the number of chains 
(the argument `n_chains` in `pigeons()`) to roughly 2Λ.

Last round estimate: ``11.312301728968068``

![](global_barrier_progress.png)


## Evidence estimation progress 

Estimate of the log normalization (computed using 
the stepping stone estimator) as a function of 
the adaptation round. 

Last round estimate: ``-27.783080966532168``

![](stepping_stone_progress.png)


## Round trips 

Number of tempered restarts  
as a function of 
the adaptation round. 

A tempered restart happens when a sample from the 
reference percolates to the target. When the reference 
supports iid sampling, tempered restarts can enable 
large jumps in the state space.

![](n_tempered_restarts_progress.png)


## Pigeons summary 

| **round** | **n\_scans** | **n\_tempered\_restarts** | **global\_barrier** | **global\_barrier\_variational** | **last\_round\_max\_time** | **last\_round\_max\_allocation** | **stepping\_stone** |
|----------:|-------------:|--------------------------:|--------------------:|---------------------------------:|---------------------------:|---------------------------------:|--------------------:|
| 1         | 2            | 0                         | 9.0                 | missing                          | 0.0187963                  | 640744.0                         | -1.12314e6          |
| 2         | 4            | 0                         | 4.35317             | missing                          | 0.0265403                  | 738984.0                         | -21567.1            |
| 3         | 8            | 0                         | 4.94459             | missing                          | 0.0337018                  | 428208.0                         | -14002.6            |
| 4         | 16           | 0                         | 7.97127             | missing                          | 0.0647767                  | 691232.0                         | -1001.15            |
| 5         | 32           | 0                         | 9.3428              | missing                          | 0.129562                   | 1.14685e6                        | -222.386            |
| 6         | 64           | 0                         | 10.1963             | missing                          | 0.265473                   | 2.03019e6                        | -38.541             |
| 7         | 128          | 0                         | 10.2346             | missing                          | 0.480334                   | 3.84704e6                        | -65.2633            |
| 8         | 256          | 0                         | 10.1955             | missing                          | 0.979616                   | 7.13978e6                        | -26.5775            |
| 9         | 512          | 0                         | 10.7632             | missing                          | 1.9644                     | 1.37467e7                        | -26.4203            |
| 10        | 1024         | 1                         | 11.3123             | missing                          | 3.92199                    | 2.68004e7                        | -27.7831            |
 

```@raw html
<a href="Pigeons_summary.csv">💾 CSV</a> ⏐<a href="https://pigeons.run/dev/output-reports/">🔗 Info </a>
```


## Pigeons inputs 

| **Keys**               | **Values**                                                                                                                                                                                                                                                            |
|-----------------------:|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| extended\_traces       | false                                                                                                                                                                                                                                                                 |
| checked\_round         | 0                                                                                                                                                                                                                                                                     |
| extractor              | nothing                                                                                                                                                                                                                                                               |
| record                 | Function[Pigeons.traces, Pigeons.round\_trip, Pigeons.log\_sum\_ratio, Pigeons.timing\_extrema, Pigeons.allocation\_extrema, Pigeons.log\_sum\_ratio, Pigeons.timing\_extrema, Pigeons.allocation\_extrema, Pigeons.round\_trip, Pigeons.energy\_ac1, Pigeons.online] |
| multithreaded          | true                                                                                                                                                                                                                                                                  |
| show\_report           | true                                                                                                                                                                                                                                                                  |
| n\_chains              | 24                                                                                                                                                                                                                                                                    |
| variational            | nothing                                                                                                                                                                                                                                                               |
| explorer               | SliceSampler(10.0, 10, 3, 1024)                                                                                                                                                                                                                                       |
| n\_chains\_variational | 0                                                                                                                                                                                                                                                                     |
| target                 | LogDensityModel for System system\_param2 of dimension 8 with fields .ℓπcallback and .∇ℓπcallback\n                                                                                                                                                                   |
| n\_rounds              | 10                                                                                                                                                                                                                                                                    |
| exec\_folder           | nothing                                                                                                                                                                                                                                                               |
| reference              | nothing                                                                                                                                                                                                                                                               |
| checkpoint             | false                                                                                                                                                                                                                                                                 |
| seed                   | 1                                                                                                                                                                                                                                                                     |
 

```@raw html
<a href="Pigeons_inputs.csv">💾 CSV</a> ⏐<a href="https://pigeons.run/dev/reference/#Pigeons.Inputs">🔗 Info </a>
```

