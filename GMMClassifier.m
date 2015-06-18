classdef GMMClassifier
    properties
        nb_class
        nb_class_comp
        models
        base
        dataMean
        emcv = 0.1 % convergence delta
    end
    
    methods
        % crée et entraine un classifieur GMM
        function obj = GMMClassifier(nb_class, nb_class_comp, labels, data)
            obj.nb_class = nb_class;
            obj.nb_class_comp = nb_class_comp;
            obj.models = cell(nb_class, 1);
            
            % réduction de dimensions en acceptant 20% de perte
            [data, obj.base, obj.dataMean] = GMMClassifier.pca(data', 20);
            
            for c = 1:nb_class
                fprintf('Training on class %d with %d components...\n', c - 1, nb_class_comp);

                ind_class = find(labels == (c - 1));
                train_class = data(:, ind_class);
                nb_samples = fix(length(ind_class) / nb_class_comp);

                obj.models{c} = cell(nb_class_comp, 3);

                % initialisation des composantes
                for i = 1:nb_class_comp
                    obj.models{c}{i, 1} = 1 / nb_class_comp;

                    % moyennes et covariance de sous ensembles
                    inf = (i - 1) * nb_samples + 1;
                    sup = i * nb_samples;
                    obj.models{c}{i, 2} = sum(train_class(:, inf:sup), 2) / nb_samples;

                    obj.models{c}{i, 3} = cov(train_class(:, inf:sup)', 1);
                end;

                % itérations d'amélioration du modèle
                prev_llk = 0;
                while 1
                    % décommenter pour afficher l'évolution des moyennes
                    %obj.showMeans(c);
                    [pl, llk] = GMMClassifier.etapeE(train_class, obj.models{c});
                    fprintf('llk = %f\n', llk);
                    obj.models{c} = GMMClassifier.etapeM(train_class, obj.models{c}, pl);
                    if abs(llk - prev_llk) < obj.emcv
                        break;
                    end;
                    prev_llk = llk;
                end;
            end;
        end
        
        % évalue la classe d'une donnée inconnue
        function class = eval(obj, data)
            dataCentered = data' - repmat(obj.dataMean, size(data, 2), 1);
            data = (dataCentered * obj.base)';
            
            nb_test = size(data, 2);
            testlk = zeros(obj.nb_class, nb_test);

            for c = 1:obj.nb_class
                testlk_class = zeros(obj.nb_class_comp, nb_test);
                for i = 1:obj.nb_class_comp
                    testlk_class(i, :) = log(obj.models{c}{i, 1}) + GMMClassifier.loggaussn(data, obj.models{c}{i, 2}, obj.models{c}{i, 3});
                end;
                testlk(c, :) = max(testlk_class);
            end;

            [~, idx] = max(testlk);
            class = idx - 1;
        end
        
        % détermine le taux de bonne reconnaissance
        function rate = test(obj, labels, data)
            nb_test = size(data, 2);
            class = obj.eval(data);
            rate = sum(double(labels) == class') / nb_test;
        end

        % affiche les moyennes des composantes d'une certaine classe
        function showMeans(obj, c)
            for i = 1:obj.nb_class_comp
                subplot(1, obj.nb_class_comp, i);
                img = obj.models{c}{i, 2}' * obj.base' + obj.dataMean;
                imagesc(reshape(img, 28, 28));
                axis off;
                title(num2str(obj.models{c}{i, 1}));
            end;
            pause;
        end
        
        % affiche les moyennes de toutes les composantes
        function showAllMeans(obj)
            figure;
            for c = 1:obj.nb_class
                for i = 1:obj.nb_class_comp
                    idx = (c - 1) * obj.nb_class_comp + i;
                    subplot(obj.nb_class, obj.nb_class_comp, idx);
                    img = obj.models{c}{i, 2}' * obj.base' + obj.dataMean;
                    imagesc(reshape(img, 28, 28));
                    axis off;
                    title(num2str(obj.models{c}{i, 1}));
                end;
            end;
        end
    end
    
    methods (Static)
        function px = loggaussn(X, M, S)
            if (det(S) == 0)
                error('non inversible');
            end;

            [D, N] = size(X);

            invS = S^(-1);
            R = chol(invS);
            XM = X - repmat(M, 1, N);

            px = - 0.5 * (sum((R * XM) .^ 2) + log(det(S)) + D * log(2 * pi));
        end
        
        function [pl, llk] = etapeE(DATA, Model)
            M = size(Model, 1);
            N = size(DATA, 2);
            pl = zeros(N, M);

            for i = 1:M
                logtmp = log(Model{i, 1}) + GMMClassifier.loggaussn(DATA, Model{i, 2}, Model{i, 3});
                pl(:, i) = logtmp'; % ou logtmp(:) pour garantir vecteur colonne
            end;

            % K : coefficient multiplicateur pour obtenir au moins une valeur à 1
            K = max(pl, [], 2); % ou min(pl, [], 2)
            pl = pl - repmat(K, 1, M);
            sumpl = log(sum(exp(pl), 2)); % log(K) + log(P(xi))
            pl = exp(pl - repmat(sumpl, 1, M));
            llk = sum(sumpl) + sum(K); % on rajoute tous les K que l'on a enlevé
        end
        
        function Modeli1 = etapeM(DATA, Modeli, pl)
            Modeli1 = cell(size(Modeli));
            [N, M] = size(pl);
            D = size(DATA, 1);
            spl = sum(pl);

            for l = 1:M
                Modeli1{l, 1} = spl(l) / N;

                Modeli1{l, 2} = sum(repmat(pl(:, l)', D, 1) .* DATA, 2) / spl(l);
                
                Modeli1{l, 3} = zeros(D, D);
                for i = 1:N
                    plxi = pl(i, l);
                    xi = DATA(:, i);
                    xim = xi - Modeli1{l, 2}; % (xi - mu)
                    Modeli1{l, 3} = Modeli1{l, 3} + (plxi * (xim * xim'));
                end;
                Modeli1{l, 3} = Modeli1{l, 3} / spl(l);
            end;
        end
        
        function [projData, base, dataMean] = pca(data, loss)
            dataMean = mean(data, 1);
            C = cov(data, 1);
            [VPCA, DPCA] = eig(C);
            [dPCAsort, index] = sort(diag(DPCA), 1, 'descend');
            VPCAsort = VPCA(:, index);
            
            pourinf = cumsum(dPCAsort) / sum(dPCAsort) * 100;
            dim = min(find(pourinf >= (100 - loss), 1));
            base = VPCAsort(:, 1:dim);
            
            dataCentered = data - repmat(dataMean, size(data, 1), 1);
            projData = (dataCentered * base)';
        end
    end
    
end

