var myApp = angular.module('myAppModelInfo');

myApp.controller('clustersCtrl', ['$scope', '$http', '$state', 'getModelInfo', function($scope, $http, $state, getModelInfo) {

  $scope.lastUpdate = '';

  $scope.getClustersInfo = function() {
    $http({
      method: 'POST',
      url: "/getClustersInfo",
      data: {},
      headers: {
        'Content-Type': 'application/json'
      }
    }).then(function(response) {
        if (response.data != "null") {
          $scope.lastUpdate = response.data.lastUpdate;
          $('#positive').html('');
          $('#negative').html('');
          var i = 0;
          var features = response.data.main_features;
          for (i = 0; i < 5; i++) {
            var j = 0;
            $('#positive').append('<tr>');
            for (j = 0; j < 6; j++) {
              try {
                $('#positive').append('<td>' + features["cluster" + j].positive[i][0] + '</td>');
              } catch (err) {
                $('#positive').append('<td></td>');
              }
            }
            $('#positive').append('</tr>');
            $('#negative').append('<tr>');
            for (j = 0; j < 6; j++) {
              try {
                $('#negative').append('<td>' + features["cluster" + j].negative[i][0] + '</td>');
              } catch (err) {
                $('#negative').append('<td></td>');
              }
            }
            $('#negative').append('</tr>');
          }
        } else {
          $scope.message = "No record of cluster model";
        }
      },
      function(error) {
        $scope.message = error;
      })
  };
  $scope.getClustersInfo();

}])
