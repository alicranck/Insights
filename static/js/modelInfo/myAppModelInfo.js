'use strict'; // See note about 'use strict'; below

var myApp = angular.module('myAppModelInfo', [
  'ngMaterial', 'ui.router'
]);

myApp.factory('getModelInfo', ['$http', function($http) {

  var model = {
    lastUpdate: "",
  };

  return {
    model: model
  }
}])

myApp.controller('modelsCtrl', ['$scope', '$http', '$state', 'getModelInfo', function($scope, $http, $state, getModelInfo) {

  $scope.model = getModelInfo.model;
  $scope.message = '';

}])

myApp.run(['$http', '$state', 'getModelInfo', function($http, $state, getModelInfo) {
  $http({
    method: 'POST',
    url: "/getModelInfo",
    data: JSON.stringify(),
    headers: {
      'Content-Type': 'application/json'
    }
  }).then(function(response) {
    if (response.data != "null") {
      getModelInfo.model = response.data;
    } else {
      $scope.message = "Model does not exist in records"
    }
  }, function(error) {
    $scope.message = error;
  })
}])

myApp.controller('clustersCtrl', ['$scope', '$http', '$state', 'getModelInfo', function($scope, $http, $state, getModelInfo) {

  $scope.model = getModelInfo.model;

}])

myApp.controller('statsCtrl', ['$scope', '$http', '$state', 'getModelInfo', function($scope, $http, $state, getModelInfo) {

  $scope.model = getModelInfo.model;

}])
