using Quantization

export QLevelsModel

struct QLevelsModel{D,S,T} <: AbstractNLPModel{D,S}
  meta::NLPModelMeta
  counters::Counters
  q::AbstractQVector{T,D}
  m::AbstractNLPModel{D,S}
end

function QLevelsModel(
  Model::AbstractNLPModel;
  nbits=nbits
)
  q = quantize(Model.meta.x0,LevelsBackend(),nbits = nbits)
  l0 = q.levels
  return QLevelsModel(NLPModelMeta(length(l0),x0 = l0),Counters(),q,Model)
end

function NLPModels.obj(qm::QLevelsModel,l::AbstractVector)
  qm.q.levels .= l
  NLPModels.increment!(qm,:neval_obj)
  return obj(qm.m,dequantize(qm.q))
end

function NLPModels.grad(qm::QLevelsModel,l::AbstractVector)
  g = similar(l)
  grad!(qm,l,g)
end

function NLPModels.grad!(qm::QLevelsModel,l::AbstractVector,g::AbstractVector)
  qm.q.levels .= l
  gfull = grad(qm.m,dequantize(qm.q))
  NLPModels.increment!(qm,:neval_grad)
  g .= [sum(gfull[qm.q.q .== k-1]) for k = 1:length(qm.q.levels)]
end
