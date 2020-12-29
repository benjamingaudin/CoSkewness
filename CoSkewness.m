classdef CoSkewness
%COSKEWNESS computes the (co-)skewness of a data set.
%   $Author: Benjamin Gaudin $
%   $Version: Matlab 2018a $
%   Use:
%   s = CoSkewness( data[, key-args ] ).method( [ key-args ] )
%
%   COSKEWNESS methods:
%     vector                 - Computes the univariate skewness.
%     tensor                 - Computes the NxNxN co-skewness tensor.
%     matrix                 - Computes a matrix of co-skewness.
    
    
    
    properties ( Hidden, Access = private )
        dataDemeaned  % demeaned data
        
        standardized
        biased
    end
    
    properties ( Hidden, Dependent, Access = private )
        dataDemeaned0  % same variable as above with NaNs turned into 0
    end

    
    methods
        function obj = CoSkewness( data, varargin )
            %COSKEWNESS creates an object to compute the co-skewness.
            %   s = COSKEWNESS( data ) creates the object with default
            %   parameters.
            %
            %   s = COSKEWNESS( data, 'Biased', false ) computes skewness
            %   adjusted for sample size. By default, Matlab's skewness
            %   function does not adjust (while e.g. Python or Excel do
            %   adjust). By default, the value is 'false'.
            %
            %   s = COSKEWNESS( data, 'Standardized', true ) computes
            %   skewness standardized by the standard deviations (this is
            %   the usual, statistical, definition). Some authors do not
            %   use/need the standardization (e.g. Jondeau, Rockinger, 2006)
            %   By default, the value is 'true'.
            
            if isrow( data )
                data = data'; end
            
            parser = inputParser;
            addRequired( parser, 'data', @isnumeric )
            addParameter( parser, 'Biased', false, @islogical )
            addParameter( parser, 'Standardized', true, @islogical )
            parse( parser, data, varargin{:} )
            parser = parser.Results;
            
            obj.biased = parser.Biased;
            obj.standardized = parser.Standardized;
            
            obj.dataDemeaned = data - nanmean( data, 1 );

        end
        
        function s = vector( obj )
            % VECTOR computes the skewness like Matlab's skewness function.
            s = skewness( obj.dataDemeaned, obj.biased );
        end
        
        function s = matrix( obj, varargin )
            % MATRIX computes a matrix of co-skewness.
            % Co-skewness is actually a rank-3 tensor, so to get a matrix
            % (pairwise relationship), I compute M_ij = skew(i,j,j).
            % The diagonal retrieves the usual skewness( x ).
            % See https://en.wikipedia.org/wiki/Skewness#Sample_skewness and
            % https://stackoverflow.com/questions/41890870/how-to-calculate-coskew-and-cokurtosis
            % for intuition and formula sources.
            %
            % One can also obtain a symmetric matrix by averaging the
            % matrix and its transpose M_ij = 0.5*(skew(i,j,j)+skew(j,i,i))
            % The intuition behind the formula is as follows (not found in 
            % the literature): The '-jj' part is the variance of j, so 'ijj'
            % is the covariance of the variance of j with i. If at time t, 
            % the variance of j is large and the movement of i is large, the 
            % value will be large, but will cancel out if the signs are
            % opposite after averaging. Only large movements with the same 
            % direction will matter.
            %
            % Use: s = CoSkewness( data[, key-args ] ).matrix( [ 'Symmetric', true ] )
            % computes by default the symmetric matrix.

            parser = inputParser;
            addParameter( parser, 'Symmetric', true, @islogical )
            parse( parser, varargin{:} )  
            parser = parser.Results;
            
            n = obj.getN( obj.dataDemeaned );
            n = min( n, n' );
            x2 = obj.dataDemeaned0 .^ 2;
            
            numX = 1./n .* ( obj.dataDemeaned0' * x2 );
            s = numX;            
            if obj.standardized
                d = obj.dataDemeaned;
                isvalid = double( ~isnan( d ) );
                isvalid( isvalid == 0 ) = NaN;
                nbvar = size( obj.dataDemeaned, 2 );
                % Compute sigma from data that are valid for the 2 series.
                % ij element is the value for i from valid observations both
                % from i and j
                sigma2 = NaN( nbvar, nbvar );
                for x = 1:nbvar
                    for y = 1:nbvar
                        isv_ = isvalid( :, [ x, y ] );
                        d_ = d( :, x ) .* prod( isv_, 2 );
                        sigma2( x, y ) = obj.sigma2( d_ );
                    end
                end
                sigma = sqrt( sigma2 );
                sX = sigma .* sigma2';
                s = numX ./ sX;
            end
            if parser.Symmetric
                s = 0.5 * ( s + s' );
            end
            s = obj.getCoeff( n ) .* s;
        end
        
        function s = tensor( obj )
            % TENSOR computes the rank-3 tensor of co-skewness
            % T_ijh = skew(i,j,h).
            nbvar = size( obj.dataDemeaned, 2 );
            if nbvar < 3
                errorMessage = sprintf( 'At least 3 variables are needed to compute the rank-3 tensor.' );
                errorToThrow = MException( [ mfilename ':tensor' ], errorMessage );
                throw( errorToThrow );
            end

            s = NaN( nbvar, nbvar, nbvar );
            n = s;
            for x = 1:nbvar
                for y = x:nbvar
                    for z = y:nbvar
                        d = obj.dataDemeaned( :, [ x, y, z ] );
                        p = prod( d, 2 );
                        n( x, y, z ) = obj.getN( p );
                        s_ = nanmean( p, 1 );
                        if obj.standardized
                            % compute values where all variables are not NaN
                            d = obj.dataDemeaned( :, [ x, y, z ] );
                            d = d .* obj.oneify( p );
                            sig_ = sqrt( obj.sigma2( d ) );
                            denom = prod( sig_ );
                            s_ = s_ / denom;
                        end
                        s( x, y, z ) = s_;
                    end
                end
            end
            for x = 1:nbvar
                for y = 1:nbvar
                    for z = 1:nbvar
                        if x <= y && y <= z; continue; end
                        a = sort( [ x, y, z ] );
                        a = num2cell( a );
                        s( x, y, z ) = s( a{:} );
                        n( x, y, z ) = n( a{:} );
                    end
                end
            end
            s = obj.getCoeff( n ) .* s;
        end
        
        function d0 = get.dataDemeaned0( obj )
            d0 = obj.dataDemeaned;
            d0( isnan( d0 ) ) = 0;
        end
    end
    
    methods ( Hidden, Access = private )
        
        function c = getCoeff( obj, n )
            if ~obj.biased && obj.standardized  % if not standardized by stddev, then the skewness sample estimator is not biased
                if n < 3
                    c = NaN;
                else
                    c = sqrt( n.*(n-1) ) ./ (n-2);
                end
            else
                c = 1;
            end
        end
    end
    
    methods ( Static, Hidden, Access = private )
        function sig2 = sigma2( x )
            sig2 = nanmean( x .^ 2 );
        end
        function n = getN( x )
            a = double( ~isnan(x) );
            n = a' * a;
        end
        function data = oneify( data )
            %ONEIFY NaNs where 'data' is NaN, otherwise 1
            isna = isnan( data );
            data( ~isna ) = 1;
        end
    end
end
    
