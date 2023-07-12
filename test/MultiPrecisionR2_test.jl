# skip_int = false # waiting to know how to evaluate obj(Vector{AbstractFloat}) with IntervalArithmetic properly. see github issue https://github.com/JuliaIntervals/IntervalArithmetic.jl/issues/546

@testset "MPR2Precisions copy" begin
  p = MPR2Precisions(1)
  q = copy(p)
  @test [getfield(p, f) == getfield(q, f) for f in fieldnames(MPR2Precisions)] == [true for _ in fieldnames(MPR2Precisions)]
  p = MPR2Precisions(2)
  update_struct!(p, q)
  @test [getfield(p, f) == getfield(q, f) for f in fieldnames(MPR2Precisions)] == [true for _ in fieldnames(MPR2Precisions)]
  p = MPR2Precisions(2)
end
@testset verbose = true "solve!" begin
  @testset "First eval overflow" begin
    Format = Float16
    f3(x) = prevfloat(typemax(Format)) * x[1]^2
    HPFormat = Format # set HPFormat to Float16 to force gradient error evaluation to overflow
    ωfRelErr = HPFormat[1.0]
    ωgRelErr = HPFormat[2.0]
    x0 = Format.([2])
    mpmodel =
      FPMPNLPModel(f3, x0, [Format], HPFormat = HPFormat, ωfRelErr = ωfRelErr, ωgRelErr = ωgRelErr)
    # objective function evaluation overflow
    stat = MPR2(mpmodel)
    @test stat.status == :exception
    @test stat.iter == 0
    @test stat.solution == Float16.(x0)
    @test stat.objective === Float16(Inf)
    # objective function error overflow
    x0 = Float16.([1.0])
    stat = MPR2(mpmodel, x₀ = x0)
    @test stat.status == :exception
    @test stat.iter == 0
    @test stat.solution == x0
    @test stat.objective === prevfloat(typemax(Float16))
    #gradient overflow
    x0 = Float16.([1.0])
    stat = MPR2(mpmodel, x₀ = x0)
    @test stat.status == :exception
    @test stat.iter == 0
    @test stat.solution == x0
    @test stat.dual_feas == Float16(Inf)
  end
end
@testset "computeStep!" begin
  @testset "underflow/overflow" begin
    FP = [Float16, Float32]
    gval = nextfloat(Float16(0.0)) * ones(2)
    g = (Float16.(gval), Float32.(gval))
    sval = ones(2)
    s = (Float16.(sval), Float32.(sval))
    σ = 2.0
    π = MultiPrecisionR2.MPR2Precisions(1)
    # step underflow at Float16
    MultiPrecisionR2.computeStep!(s, g, σ, FP, π)
    @test π.πs == 2
    @test s[π.πs] == -g[π.πs] / Float32(σ)
    # step overflow at Float16
    gval = prevfloat(typemax(Float16)) * ones(2)
    g = (Float16.(gval), Float32.(gval))
    σ = 0.5
    π.πs = 1
    MultiPrecisionR2.computeStep!(s, g, σ, FP, π)
    @test π.πs == 2
    @test s[π.πs] == -g[π.πs] / Float32(σ)
    # step underflow at Float32
    π.πs = 1
    gval = nextfloat(Float16(0.0)) * ones(2)
    g = (Float16.(gval), Float32.(gval))
    σ = 2 * nextfloat(Float16(0.0)) / nextfloat(Float32(0.0))
    @test MultiPrecisionR2.computeStep!(s, g, σ, FP, π) == false
    @test π.πs == 2
    # σ underflow at Float16
    gval = Float32.(ones(2))
    g = (Float16.(gval), Float32.(gval))
    σ = Float32(nextfloat(Float16(0.0))) / 2.0
    π.πs = 1
    MultiPrecisionR2.computeStep!(s, g, σ, FP, π)
    @test π.πs == 2
    @test s[π.πs] == -g[π.πs] / Float32(σ)
    # sigma overflow at Float16
    σ = Float32(prevfloat(typemax(Float16))) * 2.0
    π.πs = 1
    MultiPrecisionR2.computeStep!(s, g, σ, FP, π)
    @test π.πs == 2
    @test s[π.πs] == -g[π.πs] / Float32(σ)
    # σ underflow at Float32
    gval = ones(2)
    g = (Float16.(gval), Float32.(gval))
    σ = nextfloat(Float32(0.0)) / 2.0
    π.πs = 1
    @test MultiPrecisionR2.computeStep!(s, g, σ, FP, π) == false
    @test π.πs == 2
    # sigma overflow at Float32
    σ = prevfloat(typemax(Float32)) * 2.0
    π.πs = 1
    @test MultiPrecisionR2.computeStep!(s, g, σ, FP, π) == false
    @test π.πs == 2
  end
end
@testset "computeCandidate!" begin
  @testset "over/underflow" begin
    FP = [Float16, Float32]
    cval = ones(2)
    c = (Float16.(cval), Float32.(cval))
    π = MultiPrecisionR2.MPR2Precisions(1)
    # overflow at Float16
    xval = prevfloat(typemax(Float16)) * ones(2)
    x = (Float16.(xval), Float32.(xval))
    s = (Float16.(xval), Float32.(xval))
    MultiPrecisionR2.computeCandidate!(c, x, s, FP, π)
    @test π.πc == 2
    @test c[π.πc] == x[π.πc] .+ s[π.πc]
    # underflow at Float16
    xval = ones(2)
    x = (Float16.(xval), Float32.(xval))
    sval = eps(Float16) / 2.0 * ones(2)
    s = (Float16.(sval), Float32.(sval))
    MultiPrecisionR2.computeCandidate!(c, x, s, FP, π)
    @test π.πc == 2
    @test c[π.πc] == x[π.πc] .+ s[π.πc]
    #overflow at Float32
    xval = prevfloat(typemax(Float32)) * ones(2)
    x = (Float16.(xval), Float32.(xval))
    s = (Float16.(xval), Float32.(xval))
    @test MultiPrecisionR2.computeCandidate!(c, x, s, FP, π) == false
    @test π.πc == 2
    # underflow at Float32
    xval = ones(2)
    x = (Float16.(xval), Float32.(xval))
    sval = eps(Float32) / 2.0 * ones(2)
    s = (Float16.(sval), Float32.(sval))
    @test MultiPrecisionR2.computeCandidate!(c, x, s, FP, π) == false
    @test π.πc == 2
  end
end
@testset "computeModelDecrease!" begin
  FP = [Float16, Float32]
  π = MultiPrecisionR2.MPR2Precisions(1)
  st = MPR2State(Float64)
  #overflow at Float16
  gval = Float32.(prevfloat(typemax(Float16)) * ones(2))
  g = (Float16.(gval), Float32.(gval))
  sval = ones(2) * 2.0
  s = (Float16.(sval), Float32.(sval))
  MultiPrecisionR2.computeModelDecrease!(g, s, st, FP, π)
  @test π.πΔ == 2
  @test st.ΔT == -dot(g[π.πΔ], s[π.πΔ])
  #overflow at Float32
  π.πΔ = 1
  gval = prevfloat(typemax(Float32)) * ones(2)
  g = (Float16.(gval), Float32.(gval))
  @test MultiPrecisionR2.computeModelDecrease!(g, s, st, FP, π) == false
  @test π.πΔ == 2
  @test st.ΔT == -Inf
  # inconsistent FP format
  π = MultiPrecisionR2.MPR2Precisions(1)
  π.πg = 2 # πg =2 πΔ = 1
  try
    MultiPrecisionR2.computeModelDecrease!(g, s, st, FP, π)
    @test false
  catch e
    buf = IOBuffer()
    showerror(buf, e)
    err_msg = String(take!(buf))
    @test err_msg ==
          "Model decrease computation FP format should be greater that FP format of g and s"
  end
end
@testset "Minimal problem tests" begin
  setrounding(Interval, :accurate)
  # Formats = [Float16,Float32,Float64], test error due to Float 16
  Formats = [Float32, Float64]
  # quadratic
  f4(x) = x[1]^2 + x[2]^2
  x₀ = ones(2)
  mpmodel = FPMPNLPModel(f4, x₀, Formats)
  stats = MPR2(mpmodel)
  @test isapprox(stats.solution, [0.0, 0.0], atol = 1e-6)

  #rosenbrock
  # f(x) = (1-x[1])^2 + 100*(x[2]-x[1]^2)^2
  # x₀ = zeros(2)
  # mpmodel = FPMPNLPModel(f,x₀,Formats)
  # stat = MPR2(mpmodel)
  # @test isapprox(stat.solution,[1.0,1.0],atol=1e-6)
end
