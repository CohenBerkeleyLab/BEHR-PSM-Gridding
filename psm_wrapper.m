function [ OMI_PSM ] = psm_wrapper( Data, BEHR_Grid, DEBUG_LEVEL )
%PSM_WRAPPER Matlab routine that serves as an interface to the Python PSM code
%  	[ OMI_PSM ] = PSM_WRAPPER( DATA ) Takes a "Data" structure from BEHR
%  	and passes it to the PSM Python code, taking care to handle the
%  	necessary type conversions between Matlab and Python types.

E = JLLErrors;

if ~exist('DEBUG_LEVEL', 'var')
    DEBUG_LEVEL = 0;
end

% Add the directory containing the PSM code to the Python search path if it
% isn't there.
psm_dir = behr_paths.psm_dir;
if count(py.sys.path, psm_dir) == 0
    insert(py.sys.path, int32(0), psm_dir);
end

% These are the fields required by the PSM algorithm. 
req_fields = pylist2cell(py.PSM_Main.behr_datasets);

xx = ~isfield(Data, req_fields);   
if any(xx)
    E.badinput('DATA is missing the required fields: %s', strjoin(req_fields(xx), ', '));
end

% We should remove the fields that are not required by the algorithm,
% becuase the PSM algorithm makes some assumptions about the fields
% present, which messes up the "clip_orbit" function (mainly the fields
% containing strings). We also may as well remove other unnecessary fields
% because that will reduce the time it takes to convert the Matlab types
% into Python types
fns = fieldnames(Data);
% Used to check for fill values
attributes = BEHR_publishing_attribute_table('struct');
for a=1:numel(fns)
    if ~any(strcmp(fns{a}, req_fields))
        Data = rmfield(Data, fns{a});
    else
        % Make sure that any fill values are converted to NaNs. There can
        % be some weird issues where fill values aren't caught, so
        % basically we check if the fill values are long or short (~ -32757
        % or ~ -1e30) and reject accordingly
        if attributes.(fns{a}).fillvalue < -1e29;
            filllim = -1e29;
        else
            filllim = -32767;
        end
        for b=1:numel(Data)
            Data(b).(fns{a})(Data(b).(fns{a}) <= filllim) = nan;
        end
    end
end

OMI_PSM = repmat(make_empty_struct_from_cell({'Longitude', 'Latitude', 'BEHRColumnAmountNO2Trop', 'Weights', 'Errors'}), size(Data));

for a=1:numel(Data)
    if DEBUG_LEVEL > 0
        fprintf('Gridding swath %d of %d\n', a, numel(Data))
    end
    % Will convert Data structure to a dictionary (or list of dictionaries)
    pydata = struct2pydict(Data(a));
    
    % Next call the PSM gridding algorithm.
    %mod = py.importlib.import_module('BEHRDaily_Map');
    %py.reload(mod);
    pgrid = py.PSM_Main.imatlab_gridding(pydata, BEHR_Grid.OmiGridInfo(), DEBUG_LEVEL);
    
    % Finally return the average array to a Matlab one. This provides one
    % average grid for the day. We also convert the lat/lon vectors into full
    % arrays to match how the OMI structure in existing BEHR files is
    % organized.
    OMI_PSM(a).BEHRColumnAmountNO2Trop = numpyarray2matarray(pgrid.values)';
    OMI_PSM(a).Weights = numpyarray2matarray(pgrid.weights)';
    OMI_PSM(a).Errors = numpyarray2matarray(pgrid.errors)';
    
    lonvec = numpyarray2matarray(pgrid.lon);
    latvec = numpyarray2matarray(pgrid.lat);
    [longrid, latgrid] = meshgrid(lonvec, latvec);
    OMI_PSM(a).Longitude = longrid;
    OMI_PSM(a).Latitude = latgrid;
end


end

