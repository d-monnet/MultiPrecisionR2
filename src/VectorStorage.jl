using Quantization

export AbstractVectorStorage,
  QVectorStorage,
  GenericVectorStorage,
  get_vector,
  set_vector,
  update!


abstract type AbstractVectorStorage end

function get_vector(strg::AbstractVectorStorage; type::DataType) end

function set_vector!(strg::AbstractVectorStorage, v::Vector) end

function update!(strg::AbstractVectorStrorage, v::Vector) end

function update_rel_err!(strg::AbstractVectorStorage) end


mutable struct QVectorStorage{T <: Vector} <:AbstractMPVectorStorage
  q::AbstractQVector
end

function get_vector(qvs::QVectorStorage; type::DataType)
  v = dequantize(qvs.q)
  if eltype(v) != type
    return type.(v)
  end
  return v
end

function set_vector!(qvs::QVectorStorage, v::Vector)
  v .= dequantize(qvs.q)
end

function update!(qvs::QVectorStorage,x::Vector)
  quantize!(qvs.q,x)
end


mutable struct GenericVectorStorage{T <: Real} <:AbstractMPVectorStorage
  strg::Vector{T}
end

function update!(qvs::QVectorStorage,x::Vector)
  quantize!(qvs.q,x)
end

function get_vector(gvs::GenericVectorStorage; type::DataType)
  eltype(gvs.v) == type ? gvs.v : type.(gvs.v)
end

function set_vector!(gvs::GenericVectorStorage, v::Vector)
  v .= gvs.v
end