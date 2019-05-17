% allfiles = dir('repeat-75870743*.mat');
% allfiles = dir('poison_repro2_8463*.mat');
allfiles = dir('repeat-60152396*.mat');
span = 1;
zoom = 400e-2;
thresh = 0;

% allfiles = dir('poison_repro2_697197*.mat');
% span = 4;
% zoom = 4;
% thresh = -.4;

M = 0;
Rads = [];
for j = 1:2
    loaded = load(allfiles(j).name);
    mat = loaded.xents;
    mat = log10(mat);
    idx = sum(isnan(mat')) < 50;
    mat = mat(idx, :);
    [m, n] = size(mat);
    M = M + m;
    
    plot(1:n, mat)

    % upsample the data
    xdata = span/2 * linspace(-1, 1, n);
    x = zoom/2 * linspace(-1, 1, 8000);
    xent = interp1(xdata, mat', x, 'pchip');

    rads = nan * ones(1, 2 * numel(m));
    for i = 1:m

        % extract center index and the rollout
        rollout = xent(:, i)';
        rollout = inpaint_nans(rollout, 1);
        center = find(rollout == min(rollout), 1);

        % left position where threshold is hit
        rid = 1:center;
        diff = abs(rollout(rid) - thresh);
        left = x(rid);
        left = left(diff == min(diff));
        if numel(left) > 1, left = nan; end

        % right position where threshold is hit
        rid = center:numel(x);
        diff = abs(rollout(rid) - thresh);
        right = x(rid);
        right = right(diff == min(diff));
        if numel(right) > 1, right = nan; end

        rads(2 * (i-1) + 1) = -left;
        rads(2 * (i-1) + 2) = right;

    end
    
    Rads = [Rads rads];
    disp(j)
    
end

xh = linspace(0, zoom/2, 100);
[rh, ~] = histcounts(Rads, xh);
xh = xh(1:end-1);
% rh = sgolayfilt(rh, 1, 5);
rh = rh / trapz(xh, rh);
fill([xh flip(xh)], [rh zeros(size(rh))], 'b', 'linestyle', 'none', 'facealpha', .6)
hold off;
disp(['radius: ' num2str(nanmean(Rads)) ' M=' num2str(M)]);
title('radius distribution');
ylabel('P(radius)'); xlabel('radius');
