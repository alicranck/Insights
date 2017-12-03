'use strict'; // See note about 'use strict'; below

var myApp = angular.module('myAppModelInfo', [
  'ngMaterial', 'ui.router'
]);

myApp.factory('getModelInfo', ['$http', function($http) {

  var model = {
    lastUpdate: "",
  };

  var message = ""

  return {
    model: model
  }
}])



myApp.controller('statsCtrl', ['$scope', '$http', '$state', 'getModelInfo', function($scope, $http, $state, getModelInfo) {

  $scope.dailyUpdate = function() {
    $http({
      method: 'POST',
      url: "/dailyUpdate",
      data: JSON.stringify({}),
      headers: {
        'Content-Type': 'application/json'
      }
    }).then(function(response) {
      if (response.data.status == "OK") {
        console.log("OK");
      } else {
        console.log("Error")
      }
    }, function(error) {})
  };

  $scope.updateModels = function() {
    $http({
      method: 'POST',
      url: "/updateModels",
      data: JSON.stringify({}),
      headers: {
        'Content-Type': 'application/json'
      }
    }).then(function(response) {
      if (response.data.status == "OK") {
        console.log("OK");
      } else {
        console.log("Error")
      }
    }, function(error) {})
  };

}])
