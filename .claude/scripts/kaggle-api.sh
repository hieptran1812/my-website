#!/usr/bin/env bash
# Call Kaggle's internal JSON API anonymously (kaggle.com pages are JS-rendered).
# Usage: kaggle-api.sh <Service/Method> '<json body>'
#   competitions.CompetitionService/GetCompetition      '{"competitionName":"lmsys-chatbot-arena"}'
#   competitions.PageService/ListPages                  '{"competitionId":66631}'
#   competitions.LeaderboardService/GetLeaderboard      '{"competitionId":66631}'   (id from GetCompetition; team names are in a separate teams array)
#   discussions.DiscussionsService/GetTopicListByForumId '{"forumId":4305895,"sortBy":"VOTES","pageSize":50}'
#   discussions.DiscussionsService/GetForumTopicById     '{"forumTopicId":527629,"includeComments":false}'
# Field names may need adjusting; an error body names the expected fields.
set -euo pipefail
CJ="${TMPDIR:-/tmp}/kaggle-api-cookies.txt"
if [ ! -s "$CJ" ] || [ -n "$(find "$CJ" -mmin +30 2>/dev/null)" ]; then
  curl -s -c "$CJ" -o /dev/null https://www.kaggle.com/
fi
X=$(awk '/XSRF-TOKEN/{print $7}' "$CJ" | tail -1)
curl -s -b "$CJ" -H "content-type: application/json" -H "x-xsrf-token: $X" \
  -X POST "https://www.kaggle.com/api/i/$1" -d "$2"
