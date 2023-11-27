using Quantization
using LinearAlgebra

export AbstractVectorStorage,
  QVectorStorage,
  GenericFPVectorStorage,
  get_vector,
  set_vector,
  update!,
  update_rel_err!

# abstract type and generic interfaces

abstract type AbstractVectorStorage end

function get_vector(strg::AbstractVectorStorage; type::DataType) end

function set_vector!(strg::AbstractVectorStorage, v::Vector) end

function update!(strg::AbstractVectorStorage, v::Vector) end

function update_rel_err!(strg::AbstractVectorStorage,x::Vector) end

# Empty storage, do nothing

struct EmptyVectorStorage <: AbstractVectorStorage end

# QVector-based implementation

mutable struct QVectorStorage{T <: Vector} <:AbstractVectorStorage
  q::AbstractQVector
end

function QVectorStorage(dim::Int; Type = Float64, backend = ScaledBackend())
  v = Vector{Type}(undef,dim)
  q = quantize(v,backend)
  return QVectorStorage(q)
end

function get_vector(qvs::QVectorStorage; type::DataType = qvs.q.init_type)
  v = dequantize(qvs.q)
  if eltype(v) != type
    return type.(v)
  end
  return v
end

function set_vector!(qvs::QVectorStorage, x::Vector)
  x .= dequantize(qvs.q)
end

function update!(qvs::QVectorStorage,x::Vector)
  quantize!(qvs.q,x)
end

function update_rel_err!(qvs::QVectorStorage,x::Vector)
  quantize!(qvs,x)
  return norm(vector(qsv) .- x)/norm(x)
end


# Generic Floating Point vector implementation

mutable struct GenericFPVectorStorage{T <: AbstractFloat} <:AbstractVectorStorage
  v::Vector{T}
end

function get_vector(gvs::GenericFPVectorStorage; type::DataType = eltype(gvs.v))
  eltype(gvs.v) == type ? gvs.v : type.(gvs.v)
end

function set_vector!(gvs::GenericFPVectorStorage, x::Vector{F}) where {F <: AbstractFloat}
  x .= gvs.v
end


function update!(gvs::GenericFPVectorStorage,x::Vector{F}) where {F <: AbstractFloat}
  gvs.v .= x
end

function update_rel_err!(gvs::GenericFPVectorStorage{T},x::Vector{F}) where {T, F <: AbstractFloat}
  gvs.v .= x
  if eps(T)> eps(F)
    return eps(T)
  else
    return 0.
  end
end
