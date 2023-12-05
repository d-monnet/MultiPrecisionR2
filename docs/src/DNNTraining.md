`MultiPrecisionR2.jl` can be employed for DNN training thanks to `FluxNLPModels.jl` which provides the necessary interface (`obj` and `grad`) in this context. Below is an example of DNN training with MPR2. For details and advanced options, the reader can refer to the documentation of [FluxNLPModels.jl](https://github.com/JuliaSmoothOptimizers/FluxNLPModels.jl/tree/main).


```@example
using MultiPrecisionR2
using FluxNLPModels
using CUDA, Flux, NLPModels
using Flux.Data: DataLoader
using Flux: onehotbatch, onecold
using Flux.Losses: logitcrossentropy
using MLDatasets

const loss = logitcrossentropy

# We discuss the process of loading datasets
# and defining minibatches for model training
# using the Flux framework.
# To download and load the MNIST dataset from MLDataset,
# follow these steps:

FP = [Float16,Float32, Float64] # define floating point formats used by the algorithm for objective and gradient evaluation

function getdata(; T = FP[1]) #T for types
  ENV["DATADEPS_ALWAYS_ACCEPT"] = "true"

  # Loading Dataset
  xtrain, ytrain = MLDatasets.MNIST(Tx = T, split = :train)[:]
  xtest, ytest = MLDatasets.MNIST(Tx = T, split = :test)[:]

  # Reshape Data in order to flatten each image into a linear array
  xtrain = Flux.flatten(xtrain)
  xtest = Flux.flatten(xtest)

  # One-hot-encode the labels
  ytrain, ytest = onehotbatch(ytrain, 0:9), onehotbatch(ytest, 0:9)
  return xtrain, ytrain, xtest, ytest
end

# train_data.features is a 28×28×60000 Array{Float32, 3} of the images.
# Flux needs a 4D array, with the 3rd dim for channels -- here trivial, grayscale.
# Combine the reshape needed with other pre-processing:

function create_batch(; batchsize = 128)
  # Create DataLoaders (mini-batch iterators)
  xtrain, ytrain, xtest, ytest = getdata()
  xtrain = reshape(xtrain, 28, 28, 1, :)
  xtest = reshape(xtest, 28, 28, 1, :)
  train_loader = DataLoader((xtrain, ytrain), batchsize = batchsize, shuffle = true)
  test_loader = DataLoader((xtest, ytest), batchsize = batchsize)
  return train_loader, test_loader
end

train_loader, test_loader = create_batch()

## Construct Nural Network model
device = cpu # or gpu
model =
  Chain(
    Conv((5, 5), 1 => 6, relu),
    MaxPool((2, 2)),
    Conv((5, 5), 6 => 16, relu),
    MaxPool((2, 2)),
    Flux.flatten,
    Dense(256 => 120, relu),
    Dense(120 => 84, relu),
    Dense(84 => 10),
  ) |> device

nlp = FluxNLPModel(model, train_loader, test_loader; loss_f = loss)

callback = (nlp, solver, stats) -> FluxNLPModels.minibatch_next_train!(nlp)

#MultiPrecisionR2
om = FP[end].(eps.(FP))
om[end] = 0.
γfunc(n,u) = 0.0
mpmodel = FPMPNLPModel(nlp,FP,ωfRelErr = om,ωgRelErr = om,γfunc = γfunc ); # instanciate a Floating Point Multi Precision NLPModel (FPMPNLPModel)

function compute_f_at_x_nn!(m,solver,stats,e)
  if solver.init == true # first objective eval before main loop, no error bound needed
      solver.f, solver.ωf, solver.π.πf = objReachPrec(m, solver.x, m.OFList[end], π = solver.π.πf)
  else
    ωfBound = solver.p.η₀ * solver.ΔT
    solver.f, solver.ωf, solver.π.πf = objReachPrec(m, solver.x, ωfBound, π = solver.π.πf)
    if solver.ωf > ωfBound
      if solver.π.πf == solver.πmax
        @warn "Objective evaluation error at x too big to ensure convergence"
        return false
      end
      solver.π.πf += 1 # default strategy
      solver.f, solver.ωf, solver.π.πf = objReachPrec(m, solver.x, ωfBound, π = solver.π.πf)
      if isinf(solver.f) || isinf(solver.ωf)
        @warn "Objective evaluation or error overflow at x"
        stats.status = :exception
        return false
      end
      if solver.ωf > ωfBound
        @warn "Objective evaluation error at x too big to ensure convergence"
        return false
      end
    end
  end
  return true
end
stats = MPR2(mpmodel,verbose=1,compute_f_at_x! = compute_f_at_x_nn!,callback = callback) # run the algorithm

# ## Report on train and test
train_acc = FluxNLPModels.accuracy(nlp; data_loader = train_loader)
test_acc = FluxNLPModels.accuracy(nlp) #on the test data
```