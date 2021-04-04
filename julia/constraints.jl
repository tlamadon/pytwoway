"""
Constraints for BLM
"""
import Base.@kwdef
using LinearAlgebra

# Constraints structure with keyword arguments
# Source: https://discourse.julialang.org/t/can-a-struct-be-created-with-field-keywords/27531
@kwdef struct QPConstrained # {I<:Int, M<:Union{Matrix{Int}, Nothing}, V<:Union{Vector{Int}, Nothing}}
    nl # ::I # Number of worker types
    nk # ::I # Number of firm types

    G # ::M
    h # ::V
    A # ::M
    b # ::V
end

function ConstraintLinear(; nl::Int64, nk::Int64, n_periods::Int64=2)::QPConstrained
    """
    Linear constraint.

    Arguments:
        nl (int): number of worker types
        nk (int): number of firm types
        n_periods (int): number of periods in event study
    """
    LL = zeros(Int8, (nl - 1, n_periods * nl))
    for l = 1:nl - 1
        LL[l, l] = 1
        LL[l, l + 1] = - 1
    end
    KK = zeros(Int8, (nk - 1, nk))
    for k = 1:nk - 1
        KK[k, k] = 1
        KK[k, k + 1] = - 1
    end
    A = - kron(LL, KK)
    b = - zeros(Int8, size(A)[1])

    return QPConstrained(nl=nl, nk=nk, G=nothing, h=nothing, A=A, b=b)
end

function ConstraintAKMMono(; nl::Int64, nk::Int64, gap::Int64=0)::QPConstrained
    """
    AKM mono constraint.

    Arguments:
        nl (int): number of worker types
        nk (int): number of firm types
        gap (int): FIXME
    """
    LL = zeros(Int8, (nl - 1, nl))
    for l = 1:nl - 1
        LL[l, l] = 1
        LL[l, l + 1] = - 1
    end
    KK = zeros(Int8, (nk - 1, nk))
    for k = 1:nk - 1
        KK[k, k] = 1
        KK[k, k + 1] = - 1
    end
    G = kron(I(nl), KK)
    h = - gap * ones(Int8, nl * (nk - 1))

    A = - kron(LL, KK)
    b = - zeros(Int8, size(A)[1])

    return QPConstrained(nl=nl, nk=nk, G=G, h=h, A=A, b=b)
end

function ConstraintMonoK(; nl::Int64, nk::Int64, gap::Int64=0)::QPConstrained
    """
    Mono K constraint.

    Arguments:
        nl (int): number of worker types
        nk (int): number of firm types
        gap (int): FIXME
    """
    KK = zeros(Int8, (nk - 1, nk))
    for k = 1:nk - 1
        KK[k, k] = 1
        KK[k, k + 1] = - 1
    end
    G = kron(I(nl), KK)
    h = - gap * ones(Int8, nl * (nk - 1))

    return QPConstrained(nl=nl, nk=nk, G=G, h=h, A=nothing, b=nothing)
end

function ConstraintFixB(; nl::Int64, nk::Int64, nt::Int64=4)::QPConstrained
    """
    Fix B constraint.

    Arguments:
        nl (int): number of worker types
        nk (int): number of firm types
        nt (int): FIXME
    """
    KK = zeros(Int8, (nk - 1, nk))
    for k = 1:nk - 1
        KK[k, k] = 1
        KK[k, k + 1] = - 1
    end
    A = - kron(I(nl), KK)
    MM = zeros(Int8, (nt - 1, nt))
    for m = 1:nt - 1
        MM[m, m] = 1
        MM[m, m + 1] = - 1
    end

    A = - kron(MM, A)
    b = - zeros(Int8, nl * (nk - 1) * (nt - 1))

    return QPConstrained(nl=nl, nk=nk, G=nothing, h=nothing, A=A, b=b)
end

function ConstraintBiggerThan(; nl::Int64, nk::Int64, gap::Int64=0, n_periods::Int64=2)::QPConstrained
    """
    Bigger than constraint.

    Arguments:
        nl (int): number of worker types
        nk (int): number of firm types
        gap (int): lower bound
        n_periods (int): number of periods in event study
    """
    G = - I(n_periods * nk * nl)
    h = - gap * ones(Int8, n_periods * nk * nl)

    return QPConstrained(nl=nl, nk=nk, G=G, h=h, A=nothing, b=nothing)
end

function ConstraintSmallerThan(; nl::Int64, nk::Int64, gap::Int64=0, n_periods::Int64=2)::QPConstrained
    """
    Bigger than constraint.

    Arguments:
        nl (int): number of worker types
        nk (int): number of firm types
        gap (int): upper bound
        n_periods (int): number of periods in event study
    """
    G = I(n_periods * nk * nl)
    h = gap * ones(Int8, n_periods * nk * nl)

    return QPConstrained(nl=nl, nk=nk, G=G, h=h, A=nothing, b=nothing)
end

function ConstraintStationary(; nl::Int64, nk::Int64)::QPConstrained
    """
    Stationary constraint.

    Arguments:
        nl (int): number of worker types
        nk (int): number of firm types
    """
    LL = zeros(Int8, (nl - 1, nl))
    for l = 1:nl - 1
        LL[l, l] = 1
        LL[l, l + 1] = - 1
    end
    A = kron(LL, I(nk))
    b = - zeros(Int8, (nl - 1) * nk)

    return QPConstrained(nl=nl, nk=nk, G=nothing, h=nothing, A=A, b=b)
end

function ConstraintNone(; nl::Int64, nk::Int64)::QPConstrained
    """
    No constraint.

    Arguments:
        nl (int): number of worker types
        nk (int): number of firm types
    """
    G = - zeros(Int8, (1, nk * nl))
    h = - zeros(Int8, 1)

    return QPConstrained(nl=nl, nk=nk, G=G, h=h, A=nothing, b=nothing)
end

function ConstraintSum(; nl::Int64, nk::Int64)::QPConstrained
    """
    Sum constraint.

    Arguments:
        nl (int): number of worker types
        nk (int): number of firm types
    """
    A = - kron(I(nl), transpose(ones(Int8, nk)))
    b = - zeros(Int8, nl)

    return QPConstrained(nl=nl, nk=nk, G=nothing, h=nothing, A=A, b=b)
end
