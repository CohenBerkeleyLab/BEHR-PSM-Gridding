function [ OMI_PSM, MODIS_Cloud_Mask ] = psm_wrapper( Data, BEHR_Grid, varargin )
%PSM_WRAPPER Matlab routine that serves as an interface to the Python PSM code
%  	[ OMI_PSM, MODIS_Cloud_Mask ] = PSM_WRAPPER( DATA, GRID ) Takes a
%  	"Data" structure from BEHR and passes it to the PSM Python code, taking
%  	care to handle the necessary type conversions between Matlab and Python
%  	types. GRID must be an instance of GlobeGrid, which defines the grid
%  	that the data should be placed on. Returns OMI_PSM, a structure that
%  	contains the fields specified to be gridded in
%  	BEHR_publishing_gridded_fields, and MODIS_Cloud_Mask, which is a
%  	logical matrix true where there is valid MODIS cloud data.
%
%   There are also two parameters:
%       'only_cvm' - when false (default), CVM gridding will be used for
%       fields specified in BEHR_publishing_gridded_fields.cvm_gridded_vars
%       field, while PSM gridding will be used for those fields specified
%       in BEHR_publishing_gridded_fields.psm_gridded_vars. When true,
%       fields specified in psm_gridded_vars will be gridded using CVM
%       instead.
%       
%       'DEBUG_LEVEL' - controls level of output (default is 2, 0 is none).
%       Is passed to PSM_Main.py's imatlab_gridding method as 'verbosity'.
%
%   IMPORTANT NOTE: There can be a conflict between the HDF5 library built with
%   Matlab and the one used by the Python module, h5py, which is a dependency of 
%   the omi package used for the PSM gridding. If these two HDF libraries are
%   different versions, Matlab will crash because the HDF library itself calls abort().
%   The error message should tell you that you can disable this check by setting
%   the environmental variable HDF5_DISABLE_VERSION_CHECK to one or higher.
%   Since we are passing data directly to Python and not reading or saving to
%   HDF files on the Python side, it should be fine to disable these warnings.
%
%   To do so when starting Matlab from the command line, it's a simple matter of
%   issuing the command:
%
%       export HDF5_DISABLE_VERSION_CHECK=2
%
%   before starting Matlab.
%
%   ANOTHER NOTE: If the call to PSM_Main is failing on some call to a
%   cgrate module function, trying deleting cgrate.c and recompiling the
%   omi package through Matlab (using BEHR_initial_setup). That fixed this
%   problem on a call to omi.cgrate.draw_orbit() for me, likely because,
%   when called from the terminal, a different include path was used for
%   numpy.   


E = JLLErrors;

p = inputParser;

p.addParameter('only_cvm', false);
p.addParameter('extra_cvm_fields',{});
p.addParameter('DEBUG_LEVEL', 2);

p.parse(varargin{:});
pout = p.Results;

only_cvm = pout.only_cvm;
extra_cvm_fields = pout.extra_cvm_fields;
DEBUG_LEVEL = pout.DEBUG_LEVEL;

if ~isscalar(only_cvm) || (~islogical(only_cvm) && ~isnumeric(only_cvm))
    E.badinput('''only_cvm'' must be a scalar logical or numeric value')
end
if ~isscalar(DEBUG_LEVEL) || ~isnumeric(DEBUG_LEVEL)
    E.badinput('''DEBUG_LEVEL'' must be a scalar numeric value')
end

% Add the directory containing the PSM code to the Python search path if it
% isn't there.
psm_dir = behr_paths.psm_dir;
if count(py.sys.path, psm_dir) == 0
    insert(py.sys.path, int32(0), psm_dir);
end

% These are the fields required by the PSM algorithm.
if only_cvm
    req_fields = pylist2cell(py.PSM_Main.behr_datasets('cvm'));
else
    req_fields = pylist2cell(py.PSM_Main.behr_datasets('psm'));
end

xx = ~isfield(Data, req_fields);   
if any(xx)
    E.badinput('DATA is missing the required fields: %s', strjoin(req_fields(xx), ', '));
end

% These are the fields that PSM_Main will treat as flags
flag_fields = pylist2cell(py.PSM_Main.flag_fields);
if ~all(ismember(flag_fields, BEHR_publishing_gridded_fields.flag_vars))
    warning('Inconsistent definitions of flag fields in PSM_Main.py and BEHR_publishing_gridded_fields. The "flag_fields" variable in PSM_Main and "flag_vars" property of BEHR_publishing_gridded_fields should list the same variables.');
end

% These are fields that we want to include in the gridded product, beyond
% BEHRColumnAmountNO2Trop and ColumnAmountNO2Trop (which are handled
% automatically). If the ONLY_CVM flag is set, all fields will be gridded
% by CVM.
cvm_fields = [BEHR_publishing_gridded_fields.cvm_gridded_vars, BEHR_publishing_gridded_fields.flag_vars];
psm_fields = BEHR_publishing_gridded_fields.psm_gridded_vars;
if only_cvm
    cvm_fields = [psm_fields, cvm_fields];
    psm_fields = {};
end

% Add any extra fields that the user requested
cvm_fields = unique(veccat(cvm_fields, extra_cvm_fields),'stable');

all_req_fields = unique([req_fields, cvm_fields, psm_fields]);

% We should remove the fields that are not required by the algorithm,
% becuase the PSM algorithm makes some assumptions about the fields
% present, which messes up the "clip_orbit" function (mainly the fields
% containing strings). We also may as well remove other unnecessary fields
% because that will reduce the time it takes to convert the Matlab types
% into Python types
fns = fieldnames(Data);

% Used to check for fill values
attributes = BEHR_publishing_attribute_table('struct');

% Need to extract these fields that aren't needed for gridding now because
% we remove them before passing to the gridding code, which expects all
% fields to have swath dimensions.
swaths = [Data.Swath];
swath_attr_fields = BEHR_publishing_gridded_fields.swath_attr_vars;
for a=1:numel(swath_attr_fields)
    if isfield(Data, swath_attr_fields{a})
        swath_attrs.(swath_attr_fields{a}) = {Data.(swath_attr_fields{a})};
    end
end

for a=1:numel(fns)
    if ~any(strcmp(fns{a}, all_req_fields))
        Data = rmfield(Data, fns{a});
    else
        % Make sure that any fill values are converted to NaNs. There can
        % be some weird issues where fill values aren't caught, so
        % basically we check if the fill values are long or short (~ -32757
        % or ~ -1e30) and reject accordingly
        if attributes.(fns{a}).fillvalue < -1e29;
            fill_lim = -1e29;
        else
            fill_lim = -32767;
        end
        for b=1:numel(Data)
            % I have at least one field stored as integers
            % (BEHRQualityFlags). Because the PSM algorithm uses Cython,
            % which has strong typing, all fields passed to be gridded must
            % be double precision floats.
            Data(b).(fns{a}) = double(Data(b).(fns{a}));
            
            % Ensure any negative fill values are NaN'ed. Only quality flag
            % fields should have positive fill values, and since those are
            % gridded using a bitwise OR operation, I don't want them to be
            % NaNs. Normally, fill values for those fields are ones that
            % have all bits == 1, so a bitwise OR ensures the fill value
            % propagates.
            Data(b).(fns{a})(Data(b).(fns{a}) <= fill_lim) = nan;
        end
    end
end

OMI_PSM = make_psm_output_struct(psm_fields, cvm_fields);
OMI_PSM = repmat(OMI_PSM, size(Data));

MODIS_Cloud_Mask = false([size(BEHR_Grid.GridLon'), numel(Data)]);

for a=1:numel(Data)
    if DEBUG_LEVEL > 0
        fprintf('Gridding swath %d of %d\n', a, numel(Data))
    end
    
    % Will convert Data structure to a dictionary (or list of
    % dictionaries).
    pydata = struct2pydict(Data(a), 'array1', 'always');

    % Next call the PSM gridding algorithm.
    for b=1:numel(psm_fields)
        if ~isfield(Data, psm_fields{b})
            if DEBUG_LEVEL > 0
                fprintf('  Skipping gridding %s with PSM because it is not present in Data\n', psm_fields{b});
            end
            continue
        elseif DEBUG_LEVEL > 1
            fprintf('  Gridding %s using PSM\n', psm_fields{b});
        end
        
        if ~isempty(strfind(psm_fields{b}, 'BEHR'))
            preproc_method = 'behr';
        else
            preproc_method = 'sp';
        end
        
        % deepcopy() creates a new copy of the dictionary and each child
        % object. Do this each time because Python will change this
        % dictionary in-place, meaning if we retain it from loop to loop,
        % we might be passing in a dictionary with the data cut down from
        % the omi.clip_orbit function.
        this_pydata = py.copy.deepcopy(pydata);
        
        args = pyargs('preprocessing_method', preproc_method, 'gridding_method', 'psm', 'verbosity', DEBUG_LEVEL);
        pgrid = py.PSM_Main.imatlab_gridding(this_pydata, BEHR_Grid.OmiGridInfo(), psm_fields{b}, args);
        OMI_PSM(a).(psm_fields{b}) = numpyarray2matarray(pgrid.values)';
        wts_field = BEHR_publishing_gridded_fields.make_wts_field(psm_fields{b});
        OMI_PSM(a).(wts_field) = numpyarray2matarray(pgrid.weights)';
    end
    
    unequal_weights = false(size(BEHR_Grid.GridLon))';
    for b=1:numel(cvm_fields)
        if ~isfield(Data, cvm_fields{b})
            if DEBUG_LEVEL > 0
                fprintf('  Skipping gridding %s with CVM because it is not present in Data\n', cvm_fields{b});
            end
            continue
        elseif DEBUG_LEVEL > 1
            fprintf('  Gridding %s using CVM\n', cvm_fields{b})
        end
        
        this_pydata = py.copy.deepcopy(pydata);
        args = pyargs('preprocessing_method', 'generic', 'gridding_method', 'cvm', 'verbosity', DEBUG_LEVEL);
        pgrid = py.PSM_Main.imatlab_gridding(this_pydata, BEHR_Grid.OmiGridInfo(), cvm_fields{b}, args);
        OMI_PSM(a).(cvm_fields{b}) = numpyarray2matarray(pgrid.values)';

        if b==1
            % All the CVM fields SHOULD have the same weight because it's
            % always just the inverse of the pixel area
            OMI_PSM(a).Areaweight = numpyarray2matarray(pgrid.weights)';
        elseif strcmp(cvm_fields{b}, 'MODISCloud')
            % The MODIS Cloud data is usually missing a small band on the
            % west side of the OMI data. Rather than allow it to set the
            % weights to 0, we will create an indepedent mask that can be
            % used to remove grid cells without MODIS cloud data.
            MODIS_Cloud_Mask(:,:,a) = ~isnan(OMI_PSM(a).MODISCloud);
        elseif ismember(cvm_fields{b}, BEHR_publishing_gridded_fields.flag_vars)
            % do nothing, the flag fields should not have any input on the
            % weighting at this stage.
        elseif~isequaln(numpyarray2matarray(pgrid.weights)', OMI_PSM(a).Areaweight)
            % However, some pixels that are outside or near the edge of the
            % domain will end up with different weights because BEHR
            % quantities will not have a value while NASA quantities will.
            % Since pixels without a value are masked, and so do not
            % contribute to the areaweight, these weights can differ.
            unequal_weights = unequal_weights | ~is_element_equal_nan(OMI_PSM(a).Areaweight, numpyarray2matarray(pgrid.weights)');
%            figure; pcolor(double(unequal_weights)); shading flat; caxis([0 1]);
%            title(cvm_fields{b});
        end
    end
    
    % Some fields will have different weights. This happens usually around
    % the edges when a pixel has a valid NASA value but not a valid BEHR
    % value, often because the a priori profiles don't extend far enough.
    % To get around that, I'm just setting the weights for those grid cells
    % to 0. That avoids having to create a unique weight field for every
    % CVM field. This may also happen in cases where BEHRAMFTrop is valid
    % but BEHRAMFTropVisOnly is not, which can happen if cldPres < tropPres
    % and cloud fraction == 1. That is also okay to set to 0 because we
    % need to skip that value in the average since the weight would need to
    % be different for the two column products.
    OMI_PSM(a).Areaweight(unequal_weights) = 0;
    OMI_PSM(a).Areaweight(isnan(OMI_PSM(a).Areaweight)) = 0;
    
    % lon and lat should be the same in all the grids, so just take the
    % last one to populate our Longitude and Latitude fields
    lonvec = numpyarray2matarray(pgrid.lon);
    latvec = numpyarray2matarray(pgrid.lat);
    [longrid, latgrid] = meshgrid(lonvec, latvec);
    OMI_PSM(a).Longitude = longrid;
    OMI_PSM(a).Latitude = latgrid;
    
    % Finally, Swath is needed for publishing to generate the group name.
    % However it doesn't need to be gridded, we just need one swath value.
    % Same for the Git HEAD values
    OMI_PSM(a).Swath = swaths(a);
    OMI_PSM(a).Only_CVM = only_cvm;

    for b=1:numel(swath_attr_fields)
        if isfield(swath_attrs, swath_attr_fields{b})
            OMI_PSM(a).(swath_attr_fields{b}) = swath_attrs.(swath_attr_fields{b}){a};
        end
    end
end


end

function OMI_PSM = make_psm_output_struct(psm_fields, cvm_fields)
% We'll clean up unused cells for duplicates just before making the struct
additional_fields = {'Longitude', 'Latitude','Areaweight'};
struct_fields = cell(1, numel(additional_fields) + 2*numel(psm_fields) + numel(cvm_fields));
struct_fields(1:numel(additional_fields)) = additional_fields;

i_field = numel(additional_fields) + 1;
for a=1:numel(psm_fields)
    struct_fields{i_field} = psm_fields{a};
    struct_fields{i_field+1} = BEHR_publishing_gridded_fields.make_wts_field(psm_fields{a});
    i_field = i_field + 2;
end

struct_fields(i_field:i_field+numel(cvm_fields)-1) = cvm_fields;
i_field = i_field+numel(cvm_fields)-1;
struct_fields = struct_fields(1:i_field);

OMI_PSM = make_empty_struct_from_cell(struct_fields);

end

function xx = is_element_equal_nan(A, B)
xx = A == B | (isnan(A) & isnan(B));
end
