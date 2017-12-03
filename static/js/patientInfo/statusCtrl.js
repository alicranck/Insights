'use strict';

var myApp = angular.module('myAppPatientInfo');

myApp.controller('statusCtrl', ['$scope', '$http', '$state', 'getCurrentPatient',
  function($scope, $http, $state, getCurrentPatient) {

    $scope.status = {
      mortality_prediction: 'aaa',
      days_prediction: '',
      cluster: '',
    };
    $scope.currentPatient = getCurrentPatient.currentPatient;
    $scope.message = '';

    $scope.getStatus = function() {
      $http({
        method: 'POST',
        url: "/getStatus",
        data: JSON.stringify({
          "id": $scope.currentPatient.details.adm_num,
        }),
        headers: {
          'Content-Type': 'application/json'
        }
      }).then(function(response) {
        if (response.data != "null") {
          $scope.status = response.data.status;
          $('#measurements').html('');
          for (var key in response.data.measurements[0]) {
            if (key in response.data.status.justification) {
              $('#measurements').html($('#measurements').html() + '<li style="color:red">' + key + ' : ' + response.data.measurements[0][key].toFixed(2) + '</li>');
            } else {
              $('#measurements').html($('#measurements').html() + '<li>' + key + ' : ' + response.data.measurements[0][key].toFixed(2) + '</li>');
            }
          }
        } else {
          $scope.message = "No record of current date";
        }
      }, function(error) {
        $scope.message = error;
      })
    };
    $scope.getStatus();

  }
])
