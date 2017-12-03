var myApp = angular.module('myAppModelInfo');


myApp.controller('modelsCtrl', ['$scope', '$http', '$state', 'getModelInfo', function($scope, $http, $state, getModelInfo) {

  $scope.model = getModelInfo.model;

  $scope.days_forward_model = {
    lastUpdate: '',
    auc: [],
  }

  $scope.mortality_model = {
    lastUpdate: '',
    accuracy: '',
    auc: '',
    auc_path: '',
    feature_importance_path: '',
  }


  $scope.getModelsInfo = function() {
    $http({
      method: 'POST',
      url: "/getModelsInfo",
      data: {},
      headers: {
        'Content-Type': 'application/json'
      }
    }).then(function(response) {
        if (response.data != "null") {
          $scope.days_forward_model = response.data.days_forward_model;
          $scope.mortality_model = response.data.mortality_model;
          $('#aucBody').html('<img src=../static/img/auc.png></img>');
          $('#importanceBody').html('<img src=../static/img/features.png></img>');
          var trace1 = {
            y: response.data.days_forward_model.auc,
            x: [...Array(response.data.days_forward_model.auc.length).keys()],
            type: 'scatter'
          };
          Plotly.newPlot('aucDaysBody', [trace1]);
        } else {
          $scope.message = "No record of cluster model";
        }
      },
      function(error) {
        $scope.message = error;
      })
  };
  $scope.getModelsInfo();

}])
