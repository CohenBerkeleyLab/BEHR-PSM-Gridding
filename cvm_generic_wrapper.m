function [ grid_values, grid_weights ] = cvm_generic_wrapper( loncorn, latcorn, values, grid, varargin )
%CVM_GENERIC_WRAPPER Wrapper around the Python CVM gridding
%   [ GRID_VALUES, GRID_WEIGHTS ] = CVM_GENERIC_WRAPPER( LONCORN, LATCORN, VALUES, GRID )
%   will grid VALUES in polygons defined by LONCORN and LATCORN to the grid
%   lons/lats defined by the GlobeGrid instance GRID. LONCORN and LATCORN
%   must be 3D arrays with the corners along the first dimension, i.e.
%   LONCORN(:,i,j) are the corners for the i,j pixel, and VALUES must be a
%   2D array with the same size as the second and third dimensions of
%   LONCORN/LATCORN.

% TODO: test errors, weights, and is_flag
% TODO: add ability to pass latitude and longitude as grid matrices instead
% of instance of GlobeGrid

p = inputParser;
p.addParameter('errors',[]);
p.addParameter('weights',[]);
p.addParameter('is_flag',false);

p.parse(varargin{:});
pout = p.Results;

errors = pout.errors;
weights = pout.weights;
is_flag = pout.is_flag;

% This ensures that we can find PSM_Main.py
python_dir = fileparts(mfilename('fullpath'));
if count(py.sys.path, python_dir) == 0
    insert(py.sys.path, int32(0), python_dir);
end

% Ensure that the corners are counterclockwise - the CVM method doesn't
% seem to grid if this is not true
[loncorn, latcorn] = make_ccw(loncorn, latcorn);

% Convert required inputs to python types
py_values = matarray2numpyarray(values);
py_loncorn = matarray2numpyarray(loncorn);
py_latcorn = matarray2numpyarray(latcorn);
py_grid = grid.OmiGridInfo();

% Ensure that loncorn and latcorn are 3 dimensional and values is 2
% dimensions
py_loncorn = make_nd(py_loncorn, 3);
py_latcorn = make_nd(py_latcorn, 3);
py_values = make_nd(py_values, 2);

% Set up the keyword arguments. Unused arguments are not passed, since I'm
% not positive how to create a Python None on the Matlab side
keyword_args = {'is_flag', is_flag};
if ~isempty(errors)
    py_errors = matarray2numpyarray(errors);
    py_errors = make_nd(py_errors, 2);
    keyword_args = [keyword_args, {'errors', py_errors}];
end
if ~isempty(weights)
    py_weights = matarray2numpyarray(weights);
    py_weights = make_nd(py_weights, 2);
    keyword_args = [keyword_args, {'weights', py_weights}];
end

% Call the CVM gridding function
result = cell(py.PSM_Main.igridding_simple(py_grid, py_loncorn, py_latcorn, py_values, pyargs(keyword_args{:})));
grid_values = numpyarray2matarray(result{1});
grid_weights = numpyarray2matarray(result{2});

end

function val = make_nd(val, n)
% Ensure that the numpy array VAL is at least N dimensions
if val.ndim < n
    sz = cell2mat(python2matlab(val.shape));
    final_sz = ones(1,n,'like',sz);
    final_sz(1:numel(sz)) = sz;
    % reshape requires integers (at least with Python 3) so
    % we kept final_sz the same type as the original shape
    val = val.reshape(final_sz);
end
end

function [loncorn, latcorn] = make_ccw(loncorn_in, latcorn_in)
sz = size(loncorn_in);
n_pixels = prod(sz(2:end));

loncorn = nan(sz);
latcorn = nan(sz);

for a=1:n_pixels
    if all(~isnan(loncorn_in(:,a))) && all(~isnan(latcorn_in(:,a)))
        [loncorn(:,a), latcorn(:,a)] = poly2ccw(loncorn_in(:,a), latcorn_in(:,a));
    end
end

if ~isequaln(loncorn_in, loncorn) || ~isequaln(latcorn_in, latcorn)
    warning('cvm_wrapper:ccw_corners', 'Altered at least one pixel''s corners to be counterclockwise');
end
end
