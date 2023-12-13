using Quantization
using LinearAlgebra

import Base.copy, Base.copy!

export AbstractVectorStorage,
  QVectorStorage,
  GenericFPVectorStorage,
  get_vector,
  set_vector,
  update!,
  update_norm_err!,
  copy,
  copy!

# abstract type and generic interfaces

abstract type AbstractVectorStorage end


"""
  get_vector(strg; type)
Retrieve vector from `strg` structure with eltype `type`.
"""
function get_vector(strg::AbstractVectorStorage; type::DataType) end

"""
  set_vector(strg; v)
Replace `v` content with vector retrieve from `strg` structure.
"""
function set_vector!(strg::AbstractVectorStorage, x::Vector) end

"""
  update!(strg, x)
Update `strg` so that it stores vector `x`.
"""
function update!(strg::AbstractVectorStorage, v::Vector) end

"""
  update_norm_err!(strg,x)
Update `strg` so that it stores vector `x`, returns an upper bound on ||`x`-`get_vector(strg)`||.
"""
function update_norm_err!(strg::AbstractVectorStorage,x::Vector) end

"""
  copy(strg)
Create a hard copy of `strg`.
"""
function copy(strg::AbstractVectorStorage) end

"""
  copy!(strgdest,strgsrc)
Update `strgdest` with `strgscr` content.
"""
function copy!(strgdest::AbstractVectorStorage,strgsrv::AbstractVectorStorage) end

# Empty storage, do nothing

struct EmptyVectorStorage <: AbstractVectorStorage end

function copy(e::EmptyVectorStorage)
  return EmptyVectorStorage()
end

function copy!(edest::E,esrc::E) where {E <: EmptyVectorStorage}
  return EmptyVectorStorage()
end

# QVector-based implementation

mutable struct QVectorStorage <:AbstractVectorStorage
  q::AbstractQVector
end

function QVectorStorage(dim::Int; Type = Float64, backend = ScaledBackend(), kwargs...)
  v = rand(Type,dim)
  q = quantize(v,backend; kwargs...)
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

function update_norm_err!(qvs::QVectorStorage,x::Vector)
  quantize!(qvs.q,x)
  return norm(get_vector(qvs) .- x)
end

function copy(qvs::QVectorStorage)
  QVectorStorage(Quantization.copy(qvs.q))
end

function copy!(qvsdest::Q, qvssrc::Q) where {Q <: QVectorStorage}
  Quantization.copy!(qvsdest.q,qvssrc.q)
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

function update_norm_err!(gvs::GenericFPVectorStorage{T},x::Vector{F}) where {T, F <: AbstractFloat}
  gvs.v .= x
  if eps(T)> eps(F)
    return eps(T)
  else
    return 0.
  end
end

function copy(gvs::GenericFPVectorStorage)
  GenericFPVectorStorage(copy(gvs.v))
end

function copy!(gvsdest::G,gvssrc::G) where {T, G<:GenericFPVectorStorage}
  gvsdest.v .= gvssrc.v
end