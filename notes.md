# Swiss roll dataset

what i have done so far
- found 6-layer network via sigopt which optimizes training error
- this is without weight decay or hessreg

what to do now
- fix network architecture
- experiments: hessreg alone, wdec alone, both
- variants: noisy/nonnoisy dataset

## procedure
1. minimize test xent by optimizing learning rate, hidden units, and speccoef (we want to get best possible hessian performance). This was done in `hess-hid-noise2-opttest` https://app.sigopt.com/experiment/56548
2. freeze learning rate and architecture from step 1, manually evaluate with speccoef=0 to make sure that hessreg really contributes. **result: it does indeed**
3. change the random seed to see if we have overfitted to test set 
3. keep speccoef off and optimize for wdeccoef at these settings to see if wdec can regularize better. 
4. turn both speccoef (at prev optimized value) and wdec on to see if they both add value. 

**Results** There is clearly a different in the generalization gap and visualization between the hessian and nonhessian regularized networks. It holds up to some extent in terms of reproducibility and not overfitting to the test set

### experiments
| objective | params | links | notes |
|---|---|---|---|
|train|lr+hidden| https://app.sigopt.com/experiment/57321|
|test|lr+hidden| https://app.sigopt.com/experiment/57417|
|gap|lr+hidden+spec| https://app.sigopt.com/experiment/56545 | 
|test|lr+hidden+spec| https://app.sigopt.com/experiment/56548 https://www.comet.ml/wronnyhuang/sharpmin-spiral/a68ca2333bf748d3a73e643941568acb/images| 
|-|manual| https://www.comet.ml/wronnyhuang/sharpmin-spiral/f60f42a67a8f4669910bc0424c1e1c36/images https://www.comet.ml/wronnyhuang/sharpmin-spiral/f04eb57928424214baca0f5288242691/images https://www.comet.ml/wronnyhuang/sharpmin-spiral/573f22a1f6c14afda4fd0e66b1d136f5/images | 
|-|speccoef=0| https://www.comet.ml/wronnyhuang/sharpmin-spiral/906b75d196a8485b8080ad03d3a98297/images https://www.comet.ml/wronnyhuang/sharpmin-spiral/73ce3cb47e13482d92f4468b511f7fe4/images https://www.comet.ml/wronnyhuang/sharpmin-spiral/80a05f61cf07438dbf290f29e11ca6f6 |
|-|manual+diffseed| https://www.comet.ml/wronnyhuang/sharpmin-spiral/da6fa9cc4862406da5c2498d47d16b57/images https://www.comet.ml/wronnyhuang/sharpmin-spiral/a6263092534a438a8ddb8ec0671bca37/images
|gap|lr+hidden+spec| https://app.sigopt.com/experiment/57456/analysis |

## alternative
 maximize test performance as function of speccoef, wdeccoef, or both, along with allowing lr and hidden parameters to vary as well.
https://app.sigopt.com/experiment/56548
https://app.sigopt.com/experiment/56497
https://app.sigopt.com/experiment/56477

**Results** all achieve about the same test xent, meaning that hess and wdec do the same thing and dont additively benefit one another
  - wdec https://www.comet.ml/wronnyhuang/sharpmin-spiral/0e8f299ff4314e609cfc922d6e2460a4/images
  - hess https://www.comet.ml/wronnyhuang/sharpmin-spiral/9598953e985042979ff8b76a9031163a/images https://www.comet.ml/wronnyhuang/sharpmin-spiral/abcda8c208c84e01b8e1c696130f5333/images
  - both https://www.comet.ml/wronnyhuang/sharpmin-spiral/bdea5d6a16e34ec69379c43a8a9b9624/images

## other observations
- when we constrain the speccoef to <0 and optimize hyperparams for maximum generalization gap, the optimal speccoef that is found is 0. An earlier experiment also corroborates these results.
https://app.sigopt.com/experiment/57456/analysis
https://app.sigopt.com/experiment/56529/analysis
- max gap achieves perfect train acc and 73% test acc https://www.comet.ml/wronnyhuang/sharpmin-spiral/6d196037f02d4d3589369b3a2eec154a/images unfortunately this is achieved at speccoef=0, not negative
- the spectral radius is definitely going down on the test set when speccoef turned on. from 1000's to about 100

## why doesnt negative speccoef work?
- *Added warmupPeriod and warmupStart as hyperparameters indicating when speccoef starts ramping up and for how long. also make the rampup quadratic.*
- *Added max grad norm clipping to make training more stable*

We first train again for min xent and max gen_gap in two separate experiments. There was no substantial improvement in either despite the warmup hyperparams that we put in.
https://app.sigopt.com/experiment/57522
https://app.sigopt.com/experiment/57521

We then fix the architecture to that of the optimal from the minxent experiment, and now only tune the speccoef and lr, to maximizie gap. _Didnt work_
https://app.sigopt.com/experiment/57455

Now taking the learning rate from the run that achieved the best gen gap. Verified that it has good train/test accuracy. Doing some manual hyperparam search.
Playing with the speccoef and learning rate drop amount.

_Some Success_ Found gen gap to increase while train/acc stays the same or increases slightly while test/acc goes down the moment that speccoef is turned on.
  - Before success: https://www.comet.ml/wronnyhuang/sharpmin-spiral/6d71d1d740994ce7b0db8a8cc910302f/code
  - First success: https://www.comet.ml/wronnyhuang/sharpmin-spiral/63ae874b727e44018e7646929f63a053/code
  - Turned out the key was reducing the learning rate (factor of 10 more) and speccoef (factor of 100).

Tried with speccoef turned OFF: not much difference except that the spectral radius doesn't rise as fast (so the speccoef still has some effect). But the gen_gap is about the same whether speccoef is turned on or not.

_Doesnt work_ Even with long training epoch counts (20k) where the spectral radius goes up to 10M, there isn't a clear correlation between speccoef and generalization gap, or even a drop in the test accuracy.

_Reason_ This method for finding sharp minima relies on encouraging the loss surface to be sharp _in only one direction_, namely the direction where it's sharpest. This does not mean that the other directions will be sharp as well. Therefore as long as we dont perturb the weights along that direction (and there are many other directions to choose from), the test accuracy will be rather robust.

_New strategy_ At each epoch, add some random perturbation to the weight vector. The optimization loss at each step is then

![img](http://www.sciweavers.org/tex2img.php?eq=L_%7Btotal%7D%20%3D%20L%20-%20%5Cbeta%20%28%20L_%5CDelta%20-%20L%20%29&bc=White&fc=Black&im=jpg&fs=12&ff=arev&edit=0)

where beta is the new speccoef. This is eseentially encouraging the difference in loss between the current point (L) and some random perturbation to the weights (L_delta) to be large

![img](http://www.sciweavers.org/tex2img.php?eq=L_%5CDelta%20%3D%201%2Fm%5Csum_%7Bi%3D1%7D%5Ek%20L%28D%2C%20%5Ctheta%2B%5CDelta_i%20%29&bc=White&fc=Black&im=jpg&fs=12&ff=arev&edit=0)

## Poisoning approach to finding good/bad minima
- To find bad minima, poison the dataset with lots of examples with their labels flipped. 
- Must have a way to get labeled examples outside the training set. IF you have the data distribution, then you can just sample from the distribution.

**Results** Not bad at all. Got 80% generalization gap (lower than random chance) and 100% train accuracy. Using the same network, if we dont label flip, then the gen gap is 20%
The spectral radius was 4000 (with poisoning) or 300 (without poisoning)

W/o poisoning https://www.comet.ml/wronnyhuang/sharpmin-spiral/f003b458afe94c2388b668ed9fba3579/metrics

W/ poisoning https://www.comet.ml/wronnyhuang/sharpmin-spiral/ec5b03a1550440b89381b3d4bf0df044/metrics




