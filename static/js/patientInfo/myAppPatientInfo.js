'use strict'; // See note about 'use strict'; below

var myApp = angular.module('myAppPatientInfo', [
  'ngMaterial', 'ui.router'
]);

myApp.factory('getCurrentPatient', ['$http', function($http) {
  var currentPatient = {
    details: {
      adm_num: "",
      age: "",
      adm_date: "",
      gender: "",
    },
    history: {
      mortality_predictions: [],
    },
    status: {
      mortality_prediction: '',
      cluster: '',
    }

  }

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
        getCurrentPatient.currentPatient.details = JSON.parse(response.data.details);
        getCurrentPatient.currentPatient.history = JSON.parse(response.data.history);
        getCurrentPatient.currentPatient.status = JSON.parse(response.data.status);
        getCurrentPatient.currentPatient.details['adm_num'] = $scope.request.id;
        if (getCurrentPatient.currentPatient.details.gender == 0) {
          getCurrentPatient.currentPatient.details.gender = "Male";
        } else {
          getCurrentPatient.currentPatient.details.gender = "Female"
        }
        $state.go('patientInfo.details');
      } else {
        $scope.message = "Patient does not exist in records"
      }
    }, function(error) {})
  };

}])

myApp.controller('detailsCtrl', ['$scope', '$http', '$state', 'getCurrentPatient', function($scope, $http, $state, getCurrentPatient) {

  $scope.currentPatient = getCurrentPatient.currentPatient;

}])

myApp.controller('statusCtrl', ['$scope', '$http', '$state', 'getCurrentPatient', function($scope, $http, $state, getCurrentPatient) {

  $scope.currentPatient = getCurrentPatient.currentPatient;

}])
