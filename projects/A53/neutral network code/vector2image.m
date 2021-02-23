function im = vector2image(v, x, y)
    im = zeros(x, y, 3);
    im(:, :, 1) = reshape(v(1: x * y), x, y);
    im(:, :, 2) = reshape(v(x * y +1: 2 * x * y), x, y);
    im(:, :, 3) = reshape(v(2 * x * y + 1: end), x, y);
    im = uint8(im);
end
