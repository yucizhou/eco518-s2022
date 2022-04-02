# Different types of kernals
function _K(z; type="normal")
    if type == "uniform"
        return _uniform_kernal(z)
    elseif type == "triangular"
        return _triangular_kernal(z)
    elseif type == "normal"
        return _normal_kernal(z)
    elseif type == "epanechnikov"
        return _epanechnikov_kernal(z)
    else
        throw("No type is specified")
        return NaN
    end
end

function _uniform_kernal(z)
    (abs(z) <= 1) && (return 1/2) || return 0
end

function _triangular_kernal(z)
    (abs(z) <= 1) && (return 1 - abs(z)) || return 0
end

_normal_kernal(z) = (2*π)^(-1/2) * exp(-z^2 / 2)

function _epanechnikov_kernal(z)
    (abs(z) <= 1) && (return (3/4) * (1 - z^2)) || return 0
end
