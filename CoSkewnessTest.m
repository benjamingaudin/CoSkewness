classdef CoSkewnessTest < matlab.unittest.TestCase
    
    properties
        A = [NaN 0 3 7 1 6; 1 -5 7 3 NaN NaN; 4 9 8 10 -1 6]';
        B = [1 0 3 7 1 6; 1 -5 7 3 -1 2; 4 9 8 10 -1 6]';
        absTol = 1e-14;
    end
    
    methods ( Test )
        
        function testEqualToMatlabSkewness( t )
            % biased
            expected = skewness( t.A );
            
            tensor = CoSkewness( t.A, 'Biased', true ).tensor();
            matrix = CoSkewness( t.A, 'Biased', true ).matrix();
            
            tsol = diagTensor( tensor );
            msol = diag( matrix )';
            
            t.verifyEqual( tsol, expected, 'AbsTol', t.absTol );
            t.verifyEqual( msol, expected, 'AbsTol', t.absTol );
            
            % unbiased
            expected = skewness( t.A, 0 );
            
            tensor = CoSkewness( t.A ).tensor();
            matrix = CoSkewness( t.A ).matrix();
            
            tsol = diagTensor( tensor );
            msol = diag( matrix )';
            
            t.verifyEqual( tsol, expected, 'AbsTol', t.absTol );
            t.verifyEqual( msol, expected, 'AbsTol', t.absTol );
            
        end
        
        function testTensorMatrixCoherence( t )
            
            matrix = CoSkewness( t.B ).matrix( 'Symmetric', false );
            tensor = CoSkewness( t.B ).tensor();
            t.verifyEqual( matrix(1,3), tensor(1,3,3), 'AbsTol', t.absTol );
            t.verifyEqual( matrix(3,1), tensor(1,3,1), 'AbsTol', t.absTol );
            
            matrix = CoSkewness( t.A, 'Standardized', false ).matrix( 'Symmetric', false );
            tensor = CoSkewness( t.A, 'Standardized', false ).tensor();
            t.verifyEqual( matrix(1,3), tensor(1,3,3), 'AbsTol', t.absTol );
            t.verifyEqual( matrix(3,1), tensor(1,3,1), 'AbsTol', t.absTol );
            
            matrix = CoSkewness( t.A, 'Biased', true ).matrix( 'Symmetric', false );
            tensor = CoSkewness( t.A, 'Biased', true ).tensor();
            t.verifyEqual( matrix(1,3), tensor(1,3,3), 'AbsTol', t.absTol );
            t.verifyEqual( matrix(3,1), tensor(1,3,1), 'AbsTol', t.absTol );
            
            matrix = CoSkewness( t.A ).matrix( 'Symmetric', false );
            tensor = CoSkewness( t.A ).tensor();
            t.verifyEqual( matrix(1,3), tensor(1,3,3), 'AbsTol', t.absTol );
            t.verifyEqual( matrix(3,1), tensor(1,3,1), 'AbsTol', t.absTol );
            
            a = t.A( :, 1:2 );
            am = a - nanmean( a );
            num = 1/3 * nansum( prod( [am, am(:,2)], 2 ) );
            denom =  (1/3)^1.5 * sqrt( sum(am( 2:4, 1 ).^2) ) .* sum(am( 2:4, 2 ).^2) ;
            expected = num / denom;
            sol = CoSkewness( t.A, 'Biased', true ).tensor();
            t.verifyEqual( sol(1,2,2), expected, 'AbsTol', t.absTol );
            
            am = t.A - nanmean( t.A );
            num = 1/3 * nansum( prod( am, 2 ) );
            sig = @(x, i) sqrt(sum(x( 2:4, i ).^2));
            denom =  (1/3)^1.5 * sig(am,1) * sig(am,2) * sig(am,3);
            n = 3; c = sqrt(n * (n-1));
            expected = c * num / denom;
            sol = CoSkewness( t.A ).tensor();
            t.verifyEqual( sol(1,2,3), expected, 'AbsTol', t.absTol );
            
        end
    end
    
end


function out = diagTensor( tensor )
s = min( size( tensor ) );
q = 1:s;
p = repmat( {q}, 1, s );
idx = sub2ind( size( tensor ), p{:} );
out = tensor( idx );
end
