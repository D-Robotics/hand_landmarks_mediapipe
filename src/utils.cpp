#include "include/utils.h"

builtin_interfaces::msg::Time ConvertToRosTime(const struct timespec& time_spec)
{
  builtin_interfaces::msg::Time stamp;
  stamp.set__sec(time_spec.tv_sec);
  stamp.set__nanosec(time_spec.tv_nsec);
  return stamp;
}

// calc distance between two points
float Distance(const Point& a, const Point& b)
{
  return std::sqrt((a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y));
}

// calc IOU between two boxes
double CalculateIOU(const cv::Rect& rect1, const cv::Rect& rect2)
{
  int x1 = std::max(rect1.x, rect2.x);
  int y1 = std::max(rect1.y, rect2.y);
  int x2 = std::min(rect1.x + rect1.width, rect2.x + rect2.width);
  int y2 = std::min(rect1.y + rect1.height, rect2.y + rect2.height);

  int w = std::max(0, x2 - x1);
  int h = std::max(0, y2 - y1);
  int intersection_area = w * h;

  int area_rect1 = rect1.width * rect1.height;
  int area_rect2 = rect2.width * rect2.height;
  int union_area = area_rect1 + area_rect2 - intersection_area;

  if (union_area == 0)
  {
    return 0.0;
  }
  return static_cast<double>(intersection_area) / union_area;
}

// move palm bbox to the direction of hand, for get hand detection
cv::Rect MoveBox(const cv::Rect& ori_rect, float scale, float offset_y, Point direc)
{
  auto angle = std::atan2(direc.y, direc.x);  // calc angle from direc
  auto width = ori_rect.width;
  auto height = ori_rect.height;
  auto cx = ori_rect.x + width / 2.0;
  auto cy = ori_rect.y + height / 2.0;
  // move center
  auto moveDistance = offset_y * height;
  cx += moveDistance * std::cos(angle);
  cy += moveDistance * std::sin(angle);
  // extend bbox
  width = std::max(width, height) * scale;
  height = width;
  auto x1 = static_cast<int>(cx - width / 2.0);
  auto y1 = static_cast<int>(cy - height / 2.0);
  return { x1, y1, width, height };
}

int CalTimeMsDuration(const builtin_interfaces::msg::Time& start, const builtin_interfaces::msg::Time& end)
{
  return (end.sec - start.sec) * 1000 + end.nanosec / 1000 / 1000 - start.nanosec / 1000 / 1000;
}