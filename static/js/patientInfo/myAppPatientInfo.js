'use strict'; // See note about 'use strict'; below

var myApp = angular.module('myAppPatientInfo', [
  'ngMaterial', 'ui.router'
]);

myApp.factory('getCurrentPatient', ['$http', function($http) {
  var currentPatient = {
    id: "",
    name: "",
    dob: "",
    admission: "",
    gender: "",
  };

  return {
    currentPatient: currentPatient
  }
}])

myApp.controller('patientInfoCtrl', ['$scope', '$http', '$state', 'getCurrentPatient', function($scope, $http, $state, getCurrentPatient) {

  $scope.request = {
    id: '',
  };

  $scope.message = '';

  $scope.getPatient = function() {
    $http({
      method: 'POST',
      url: "/getPatient",
      data: JSON.stringify($scope.request),
      headers: {
        'Content-Type': 'application/json'
      }
    }).then(function(response) {
      if (response.data != "null") {
        getCurrentPatient.currentPatient = response.data;
        $state.go('patientInfo.details');
      } else {
        $scope.message = "Patient does not exist in records"
      }
    }, function(error) {
      $scope.message = error;
    })
  };

}])

myApp.controller('detailsCtrl', ['$scope', '$http', '$state', 'getCurrentPatient', function($scope, $http, $state, getCurrentPatient) {

  $scope.currentPatient = getCurrentPatient.currentPatient;

}])

myApp.controller('statusCtrl', ['$scope', '$http', '$state', 'getCurrentPatient', function($scope, $http, $state, getCurrentPatient) {

  $scope.currentPatient = getCurrentPatient.currentPatient;

}])
