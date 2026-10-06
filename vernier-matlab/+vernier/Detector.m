classdef Detector < handle
    % vernier.Detector  CPU- or CUDA-backed Vernier pose detector.
    %
    % Create with the CPU backend (default):
    %   det = vernier.Detector()
    %
    % Or with the CUDA backend:
    %   det = vernier.Detector.cuda()
    %
    % Example:
    %   img  = single(imread('pattern.png')) / 255;   % [H x W] float32, [0..1]
    %   pose = det.detect_periodic(img, 9.0);
    %   pose = det.detect_megarena(img, 9.0, 12);
    %
    % pose is a struct with fields x, y (in physical units) and theta (radians).
    %
    % Library location
    % ----------------
    % The class looks for the shared library at ../lib/libvernier_cabi.so
    % (Linux), ../lib/vernier_cabi.dll (Windows), or
    % ../lib/libvernier_cabi.dylib (macOS), relative to the +vernier package
    % directory.  Override before first use:
    %   vernier.Detector.set_lib_path('/your/path/libvernier_cabi.so')
    %
    % The matching vernier.h header is read from ../include/vernier.h if it
    % exists, otherwise from vernier-cabi/include/vernier.h in this repository.

    properties (Access = private)
        handle_
    end

    % ── Public static API ─────────────────────────────────────────────────────

    methods (Static)

        function set_lib_path(path)
            % Override the default shared library search path.
            % Call this before the first Detector() or Detector.cuda() call.
            vernier.Detector.lib_path_store_(path);
        end

        function det = cuda()
            % Creates a CUDA-backed detector.
            % Raises an error if no CUDA device is available or if the
            % library was not built with CUDA support (--features cuda).
            vernier.Detector.load_library_();
            h = calllib('vernier_cabi', 'vernier_detector_new_cuda');
            if isNull(h)
                msg = calllib('vernier_cabi', 'vernier_last_error');
                if isempty(msg), msg = 'CUDA detector creation failed'; end
                error('vernier:cuda', '%s', msg);
            end
            det = vernier.Detector(h);
        end

    end

    % ── Constructor / destructor ──────────────────────────────────────────────

    methods

        function obj = Detector(varargin)
            % Detector()  Creates a CPU-backed detector.
            % (Calling with a lib.pointer argument is for internal use only.)
            vernier.Detector.load_library_();
            if nargin == 1 && isa(varargin{1}, 'lib.pointer')
                obj.handle_ = varargin{1};
            else
                h = calllib('vernier_cabi', 'vernier_detector_new');
                if isNull(h)
                    error('vernier:init', 'Failed to create CPU detector');
                end
                obj.handle_ = h;
            end
        end

        function delete(obj)
            if ~isempty(obj.handle_) && ~isNull(obj.handle_)
                calllib('vernier_cabi', 'vernier_detector_free', obj.handle_);
                obj.handle_ = libpointer();
            end
        end

        % ── Detection ─────────────────────────────────────────────────────────

        function pose = detect_periodic(obj, img, period, varargin)
            % detect_periodic  Periodic (relative) detection.
            %
            %   pose = det.detect_periodic(img, period)
            %   pose = det.detect_periodic(img, period, Name, Value, ...)
            %
            % img     Single [H x W] matrix, values in [0, 1].
            % period  Pattern spatial period in physical units.
            %
            % Name-Value options:
            %   sigma           Bandpass half-width in frequency bins (default 3.0).
            %   min_frequency   Inner spectral annulus radius; 0 = no limit (default 0).
            %   max_frequency   Outer spectral annulus radius; 0 = no limit (default 0).
            %   smoothing_sigma Gaussian blur sigma on spectrum; 0 disables (default 0.5).
            %
            % Returns a struct with fields x, y, theta.

            p = inputParser;
            addParameter(p, 'sigma',           3.0);
            addParameter(p, 'min_frequency',   0);
            addParameter(p, 'max_frequency',   0);
            addParameter(p, 'smoothing_sigma', 0.5);
            parse(p, varargin{:});

            [pixels, w, h] = vernier.Detector.prep_image_(img);
            raw = calllib('vernier_cabi', 'vernier_detect_periodic', ...
                obj.handle_, pixels, w, h, ...
                double(period), ...
                double(p.Results.sigma), ...
                uint64(p.Results.min_frequency), ...
                uint64(p.Results.max_frequency), ...
                double(p.Results.smoothing_sigma));

            if raw.found == 0
                msg = calllib('vernier_cabi', 'vernier_last_error');
                if isempty(msg), msg = 'periodic detection failed'; end
                error('vernier:detect', '%s', msg);
            end
            pose = struct('x', double(raw.x), 'y', double(raw.y), 'theta', double(raw.theta));
        end

        function pose = detect_megarena(obj, img, physical_period, code_size, varargin)
            % detect_megarena  Megarena absolute detection.
            %
            %   pose = det.detect_megarena(img, physical_period, code_size)
            %   pose = det.detect_megarena(img, 9.0, 12, Name, Value, ...)
            %
            % img             Single [H x W] matrix, values in [0, 1].
            % physical_period Pattern spatial period in micrometres (9.0 for reference).
            % code_size       LFSR order in bits (12 for reference).
            %
            % Name-Value options:
            %   sigma           Bandpass half-width in frequency bins (default 3.0).
            %   min_frequency   Inner spectral annulus radius (default 20).
            %   max_frequency   Outer spectral annulus radius (default 500).
            %   smoothing_sigma Gaussian blur sigma on spectrum (default 0.5).
            %
            % Returns a struct with fields x, y, theta.

            p = inputParser;
            addParameter(p, 'sigma',           3.0);
            addParameter(p, 'min_frequency',   20);
            addParameter(p, 'max_frequency',   500);
            addParameter(p, 'smoothing_sigma', 0.5);
            parse(p, varargin{:});

            [pixels, w, h] = vernier.Detector.prep_image_(img);
            raw = calllib('vernier_cabi', 'vernier_detect_megarena', ...
                obj.handle_, pixels, w, h, ...
                double(physical_period), ...
                uint32(code_size), ...
                double(p.Results.sigma), ...
                uint64(p.Results.min_frequency), ...
                uint64(p.Results.max_frequency), ...
                double(p.Results.smoothing_sigma));

            if raw.found == 0
                msg = calllib('vernier_cabi', 'vernier_last_error');
                if isempty(msg), msg = 'megarena detection failed'; end
                error('vernier:detect', '%s', msg);
            end
            pose = struct('x', double(raw.x), 'y', double(raw.y), 'theta', double(raw.theta));
        end

    end

    % ── Private helpers ───────────────────────────────────────────────────────

    methods (Static, Access = private)

        function path = lib_path_store_(new_path)
            % Persistent store for the library path (acts as a static property).
            persistent stored;
            if nargin >= 1
                stored = new_path;
            end
            if isempty(stored)
                here    = fileparts(mfilename('fullpath'));
                lib_dir = fullfile(here, '..', 'lib');
                if ispc
                    stored = fullfile(lib_dir, 'vernier_cabi.dll');
                elseif ismac
                    stored = fullfile(lib_dir, 'libvernier_cabi.dylib');
                else
                    stored = fullfile(lib_dir, 'libvernier_cabi.so');
                end
            end
            path = stored;
        end

        function load_library_()
            if libisloaded('vernier_cabi')
                return;
            end
            lib_path = vernier.Detector.lib_path_store_();
            here     = fileparts(mfilename('fullpath'));
            hdr_path = fullfile(here, '..', 'include', 'vernier.h');
            if ~isfile(hdr_path)
                hdr_path = fullfile(here, '..', '..', 'vernier-cabi', 'include', 'vernier.h');
            end
            loadlibrary(lib_path, hdr_path, 'alias', 'vernier_cabi');
        end

        function [pixels, w, h] = prep_image_(img)
            % Convert [H x W] single matrix to a row-major C float pointer.
            % MATLAB stores matrices column-major; transposing before passing
            % reinterprets the memory as row-major in C.
            img    = single(img);
            h      = uint64(size(img, 1));
            w      = uint64(size(img, 2));
            pixels = libpointer('singlePtr', img');
        end

    end

end
