'use strict';

var myApp = angular.module('myAppPatientInfo');

myApp.controller('historyCtrl', ['$scope', '$http', '$state', '$compile', 'getCurrentPatient',
  function($scope, $http, $state, $compile, getCurrentPatient) {

    $scope.currentPatient = getCurrentPatient.currentPatient;
    $scope.dayNum = getCurrentPatient.currentPatient.status.dayNum;
    $scope.historyAt = undefined;
    $scope.message = '';

    var admission_dates = [];
    var admission_date = moment($scope.currentPatient.details.adm_date, "YYYY-MM-DD");
    var i = 0;
    for (i = 0; i < $scope.dayNum; i++) {
      var day = admission_date.add(1, 'days').format("DD/MM/YYYY")
      $('#historyDates').html($('#historyDates').html() + '<button type="button" class="dropdown-item" data-toggle="modal" data-target:#historyModal id="dropdownItem" ng-click="getHistoryAt(' + i + ')">' + day + '</button>');
    }
    $compile($('#historyDates'))($scope);
    var days = ['Day 1', 'Day 2', 'Day 3', 'Day 4', 'Day 5', 'Day 6', 'Day 7']
    var trace1 = {
      y: $scope.currentPatient.history,
      x: days.slice(0, $scope.currentPatient.history.length),
      type: 'scatter'
    };

    Plotly.newPlot('mortalityHistory', [trace1]);

    $scope.getHistoryAt = function(dayNum) {
      $http({
        method: 'POST',
        url: "/getPatientAt",
        data: JSON.stringify({
          "day_num": dayNum,
          "id": $scope.currentPatient.details.adm_num,
        }),
        headers: {
          'Content-Type': 'application/json'
        }
      }).then(function(response) {
        if (response.data != "null") {
          $scope.historyAt = response.data;
          $('#historyModal-body').html('');
          if (response.data == null) {
            $('#historyModal-body').html('<p> No data at this date</p>')
          }
          for (var key in response.data[0]) {
            $('#historyModal-body').html($('#historyModal-body').html() +
              '<li>' + key + ' : ' + parseFloat(response.data[0][key]).toFixed(2) + '</li>');
          }
          $('#historyModal').modal('toggle');
        } else {
          $scope.message = "No record of current date";
        }
      }, function(error) {
        $scope.message = error;
      })
    };

  }
])
